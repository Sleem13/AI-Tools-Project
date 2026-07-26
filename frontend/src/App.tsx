import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Datasets from "./pages/Datasets";
import Training from "./pages/Training";
import Settings from "./pages/Settings";
import Workflow from "./pages/Workflow";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/datasets" element={<Datasets />} />
        <Route path="/training" element={<Training />} />
        <Route path="/workflow" element={<Workflow />} />
        <Route path="/settings" element={<Settings />} />
      </Route>
    </Routes>
  );
}
