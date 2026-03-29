## Plan: OpenClaw Planner via Streaming (DRAFT)

Objetivo: integrar OpenClaw como planner de ações em streaming sem quebrar a conversa fluida do caminho atual Realtime + Reachy SDK. A estratégia mantém Realtime como camada de áudio e turn-taking, move planejamento para um worker assíncrono dedicado e aplica fallback explícito ao usuário em falhas/timeout. O desenho evita bloqueio em callbacks, introduz cancelamento por turn_id e usa a máquina de estados já existente para sinalizar timeout/model error. Isso reduz risco operacional, preserva latência pragmática de MVP e prepara evolução futura para cérebro total sem refatoração disruptiva.

**Steps**
1. Definir fronteira de planejamento com um adaptador no loop principal em [src/bridge/main.py](src/bridge/main.py#L121-L294), adicionando abstração BrainAdapter e pontos de chamada em on_user_text, on_speech_start, on_error e apply_event em [src/bridge/main.py](src/bridge/main.py#L167-L210).  
2. Criar cliente de planner em streaming (OpenClawWSClient) em novo módulo sob src/bridge (ex.: src/bridge/brain/openclaw_ws.py), com API de submit_turn, cancel_turn, callbacks parcial/final/erro e timeouts em camadas (conexão, primeiro chunk, total).  
3. Inserir fila/worker dedicado ao planner no runtime (padrão semelhante ao worker de movimento em [src/bridge/reachy/motion.py](src/bridge/reachy/motion.py#L160-L479)), garantindo que callbacks do realtime só enfileirem trabalho rápido.  
4. Implementar protocolo de cancelamento por turn_id: novo speech_started cancela planejamento em andamento, descarta eventos atrasados e evita aplicação de plano obsoleto; integrar com transições de estado em [src/bridge/state_machine.py](src/bridge/state_machine.py#L15-L63).  
5. Padronizar resultado de planejamento para execução física existente: converter plano OpenClaw em ações suportadas por ReachyClient em [src/bridge/reachy/client.py](src/bridge/reachy/client.py#L32-L87), mantendo validação defensiva de payload.  
6. Adicionar fallback explícito ao usuário em falha do planner (timeout, desconexão, payload inválido): emitir mensagem curta de erro e continuar conversa em modo local seguro; disparar Event.MODEL_ERROR ou Event.TIMEOUT na state machine em [src/bridge/state_machine.py](src/bridge/state_machine.py#L15-L23).  
7. Tornar thread safety explícito no controle de estado: serializar transições via fila de eventos do loop principal ou lock leve para evitar corrida entre thread realtime e loop principal em [src/bridge/main.py](src/bridge/main.py#L167-L173) e [src/bridge/main.py](src/bridge/main.py#L257-L260).  
8. Estender configuração com parâmetros de planner em [src/bridge/config.py](src/bridge/config.py#L18-L49): endpoint WS, timeout conexão, timeout primeiro chunk, timeout total, fallback message, toggle de planner habilitado.  
9. Atualizar dependências para suporte WS e reconexão no ambiente Python em [requirements.txt](requirements.txt), mantendo compatibilidade com execução atual.  
10. Instrumentar observabilidade de latência com logs estruturados por sessão/turno no fluxo em [src/bridge/main.py](src/bridge/main.py) e [src/bridge/reachy/realtime_client.py](src/bridge/reachy/realtime_client.py): planner_connect_ms, first_chunk_ms, final_ms, fallback_reason, cancelled.  
11. Alinhar documentação runtime vs plano de migração em [README.md](README.md), [docs/architecture.md](docs/architecture.md) e nota de contrato em [docs/api-contract.md](docs/api-contract.md) indicando uso de streaming no MVP desta etapa.

**Verification**
- Smoke local com planner desabilitado: python -m bridge.main --mode realtime (comportamento atual preservado).  
- Smoke com planner habilitado e OpenClaw disponível: validar caminho STT → planner parcial/final → execução de ação → resposta.  
- Teste de timeout (simular atraso no planner): confirmar fallback explícito ao usuário e transição para erro sem travar sessão.  
- Teste de cancelamento: iniciar turno, interromper com nova fala, verificar descarte do plano antigo e ausência de ação fantasma.  
- Teste de robustez de payload: enviar resposta WS inválida e confirmar tratamento seguro + logs de motivo.  
- Verificação de latência: inspecionar métricas de log por turno para ajustar timeouts de MVP.

**Decisions**
- OpenClaw entra agora como planner de ações, não cérebro total.  
- Integração inicial usa Streaming/WS.  
- Meta de latência é MVP pragmática (com timeouts e fallback robustos).  
- Em falha de OpenClaw, comportamento é erro explícito ao usuário com continuidade controlada.
