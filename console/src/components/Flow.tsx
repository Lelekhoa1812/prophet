import type { ReactNode } from "react";
import { ArrowDown, ArrowRight } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Flow diagrams in plain HTML and CSS so text can never clip or overflow its box. Nodes wrap, arrows turn
 * downward on narrow screens, and every node has real text, not coordinates.
 */
export function FlowNode({ title, sub, tone = "default", step }: { title: string; sub?: ReactNode; tone?: "default" | "accent" | "muted" | "warn"; step?: number }) {
  const t = {
    default: "bg-card",
    accent: "bg-accent border-primary",
    muted: "bg-menu-hover",
    warn: "bg-amber-50 border-amber-300 dark:bg-amber-950 dark:border-amber-800",
  }[tone];
  return (
    <div className={cn("min-w-0 flex-1 rounded-lg border p-3 shadow-[0_1px_2px_rgba(19,42,59,0.06)]", t)}>
      <p className="flex items-center gap-2 text-sm font-semibold leading-snug">
        {step != null && <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-primary text-[11px] font-semibold text-primary-foreground">{step}</span>}
        {title}
      </p>
      {sub && <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{sub}</p>}
    </div>
  );
}

export const FlowArrow = ({ label }: { label?: string }) => (
  <div className="flex shrink-0 items-center justify-center py-1 text-muted-foreground md:px-1 md:py-0" aria-hidden>
    <span className="flex flex-col items-center gap-0.5 md:flex-row">
      <ArrowDown className="h-4 w-4 md:hidden" />
      <ArrowRight className="hidden h-4 w-4 md:block" />
      {label && <span className="text-[10px] uppercase tracking-wide">{label}</span>}
    </span>
  </div>
);

/** A row of nodes joined by arrows; stacks vertically on phones. */
export function FlowRow({ children, className }: { children: ReactNode[]; className?: string }) {
  const items = (Array.isArray(children) ? children : [children]).flat();
  return (
    <div className={cn("flex flex-col items-stretch md:flex-row md:items-stretch", className)}>
      {items.map((c, i) => (
        <div key={i} className="flex flex-col items-stretch md:contents">
          {i > 0 && <FlowArrow />}
          {c}
        </div>
      ))}
    </div>
  );
}

/** Several inputs that all feed one node (e.g. three kinds of trigger into one agent run). */
export function FlowMerge({ inputs, children }: { inputs: ReactNode[]; children: ReactNode }) {
  return (
    <div className="flex flex-col items-stretch md:flex-row md:items-center">
      <div className="flex flex-1 flex-col gap-2">{inputs}</div>
      <FlowArrow />
      <div className="flex flex-1 flex-col md:max-w-[34%]">{children}</div>
    </div>
  );
}

/** Plain-language worked example shown under a diagram. */
export function Example({ title = "A real example from the twin", children }: { title?: string; children: ReactNode }) {
  return (
    <div className="mt-4 rounded-md border border-dashed bg-menu-hover p-3 text-sm">
      <p className="eyebrow mb-1">{title}</p>
      <div className="space-y-1 leading-relaxed">{children}</div>
    </div>
  );
}
