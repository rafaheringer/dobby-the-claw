import { BrowserRouter, Navigate, Route, Routes } from "react-router";
import Chat from "../routes/Chat";
import Home from "../routes/Home";
import Logs from "../routes/Logs";
import System from "../routes/System";
import SystemWake from "../routes/SystemWake";
import Layout from "./Layout";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="conversa" element={<Chat />} />
          <Route path="logs" element={<Logs />} />
          <Route path="sistema" element={<System />} />
          <Route path="sistema/nome" element={<SystemWake />} />
          <Route path="twin" element={null} /> {/* drawn by Layout over the home page */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
