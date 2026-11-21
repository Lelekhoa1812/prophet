import { BarChart3, ScanSearch, TrendingUp, type LucideIcon } from "lucide-react";

// Starter skills: a named, reusable way of asking Prophet for one kind of analysis. Choosing one fills the message box
// with its prompt, so the person can add detail before sending. Plain words only, no internal names.
export type Skill = { key: string; title: string; description: string; prompt: string; icon: LucideIcon };

export const SKILLS: Skill[] = [
  {
    key: "forecast-30",
    title: "Forecast the next 30 days",
    description: "A point forecast with a likely range for the main series.",
    prompt: "Forecast the next 30 days of the main series in the attached files, with a likely range.",
    icon: TrendingUp,
  },
  {
    key: "backtest",
    title: "Check forecast accuracy",
    description: "Back-test on the last 12 weeks and report the error.",
    prompt: "Back-test the forecast on the last 12 weeks of the attached data and report the error.",
    icon: BarChart3,
  },
  {
    key: "anomalies",
    title: "Spot unusual spikes or drops",
    description: "Find unusual points and explain what they may mean.",
    prompt: "Detect unusual spikes or drops in the attached data and explain what may cause them.",
    icon: ScanSearch,
  },
];
