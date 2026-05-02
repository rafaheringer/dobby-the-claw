"""Weather tool — queries Open-Meteo for current conditions and forecast."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict

import requests

from bridge.tools.contracts import ToolDefinition, ToolExecutionResult

_LOG = logging.getLogger(__name__)

_DEFAULT_LAT = -22.9068
_DEFAULT_LON = -43.1729
_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT_S = 8

_WMO_DESCRIPTIONS: dict[int, str] = {
    0: "céu limpo",
    1: "predominantemente limpo",
    2: "parcialmente nublado",
    3: "nublado",
    45: "névoa",
    48: "névoa com gelo",
    51: "garoa leve",
    53: "garoa moderada",
    55: "garoa forte",
    56: "garoa congelante leve",
    57: "garoa congelante forte",
    61: "chuva fraca",
    63: "chuva moderada",
    65: "chuva forte",
    66: "chuva congelante leve",
    67: "chuva congelante forte",
    71: "neve fraca",
    73: "neve moderada",
    75: "neve forte",
    77: "granizo",
    80: "pancadas de chuva fracas",
    81: "pancadas de chuva moderadas",
    82: "pancadas de chuva fortes",
    85: "pancadas de neve fracas",
    86: "pancadas de neve fortes",
    95: "tempestade",
    96: "tempestade com granizo leve",
    99: "tempestade com granizo forte",
}

_DAY_NAMES_PT = [
    "segunda-feira",
    "terça-feira",
    "quarta-feira",
    "quinta-feira",
    "sexta-feira",
    "sábado",
    "domingo",
]


def _wmo_description(code: int) -> str:
    return _WMO_DESCRIPTIONS.get(code, f"condição {code}")


def _safe_index(lst: list, i: int, default=None):
    return lst[i] if i < len(lst) else default


class WeatherTool:
    def __init__(self, default_city: str = "Rio de Janeiro") -> None:
        self._default_city = default_city

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_weather",
            description=(
                "Busca o clima atual e a previsão do tempo para uma cidade. "
                "Use para responder perguntas sobre o tempo hoje, amanhã ou nos próximos dias. "
                f"Padrão: {self._default_city}."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "Nome da cidade (ex: 'São Paulo', 'Curitiba'). "
                            f"Omitir usa '{self._default_city}'."
                        ),
                    },
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 7,
                        "description": (
                            "Número de dias de previsão (1=hoje, 2=hoje+amanhã, 7=semana). Padrão: 3."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        city = (arguments.get("location") or self._default_city).strip()
        days = max(1, min(7, int(arguments.get("days", 3))))

        lat, lon, resolved_name = self._geocode(city)
        if lat is None:
            return ToolExecutionResult(output={"ok": False, "error": f"Cidade não encontrada: {city}"})

        try:
            data = self._fetch_forecast(lat, lon, days)
        except Exception as exc:
            _LOG.warning("Weather fetch failed: %s", exc)
            return ToolExecutionResult(output={"ok": False, "error": "Falha ao buscar dados de clima."})

        return ToolExecutionResult(output=self._build_output(resolved_name, data))

    def _geocode(self, city: str) -> tuple[float | None, float | None, str]:
        if city.lower() in {"rio de janeiro", "rio"}:
            return _DEFAULT_LAT, _DEFAULT_LON, "Rio de Janeiro"
        try:
            resp = requests.get(
                _GEOCODING_URL,
                params={"name": city, "count": 1, "language": "pt", "format": "json"},
                timeout=_TIMEOUT_S,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                return None, None, city
            r = results[0]
            label = r.get("name", city)
            country = r.get("country", "")
            if country:
                label = f"{label}, {country}"
            return float(r["latitude"]), float(r["longitude"]), label
        except Exception as exc:
            _LOG.warning("Geocoding failed for '%s': %s", city, exc)
            return None, None, city

    def _fetch_forecast(self, lat: float, lon: float, days: int) -> dict:
        resp = requests.get(
            _FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "temperature_2m,apparent_temperature,weather_code,"
                    "wind_speed_10m,relative_humidity_2m,precipitation,is_day"
                ),
                "daily": (
                    "temperature_2m_max,temperature_2m_min,weather_code,"
                    "precipitation_sum,precipitation_probability_max"
                ),
                "timezone": "America/Sao_Paulo",
                "forecast_days": days,
            },
            timeout=_TIMEOUT_S,
        )
        resp.raise_for_status()
        return resp.json()

    def _build_output(self, location: str, data: dict) -> dict:
        cur = data.get("current", {})
        daily = data.get("daily", {})

        current = {
            "temp_c": cur.get("temperature_2m"),
            "feels_like_c": cur.get("apparent_temperature"),
            "description": _wmo_description(cur.get("weather_code", 0)),
            "wind_kmh": cur.get("wind_speed_10m"),
            "humidity_pct": cur.get("relative_humidity_2m"),
            "precipitation_mm": cur.get("precipitation"),
            "is_day": bool(cur.get("is_day", 1)),
        }

        dates = daily.get("time", [])
        forecast = []
        for i, date_str in enumerate(dates):
            try:
                day_name = _DAY_NAMES_PT[datetime.fromisoformat(date_str).weekday()]
            except Exception:
                day_name = date_str
            forecast.append(
                {
                    "date": date_str,
                    "day_of_week": day_name,
                    "max_c": _safe_index(daily.get("temperature_2m_max", []), i),
                    "min_c": _safe_index(daily.get("temperature_2m_min", []), i),
                    "description": _wmo_description(
                        _safe_index(daily.get("weather_code", []), i, 0)
                    ),
                    "rain_mm": _safe_index(daily.get("precipitation_sum", []), i),
                    "rain_probability_pct": _safe_index(
                        daily.get("precipitation_probability_max", []), i
                    ),
                }
            )

        return {
            "ok": True,
            "location": location,
            "current": current,
            "daily_forecast": forecast,
        }
