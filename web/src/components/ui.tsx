import { cva, type VariantProps } from "class-variance-authority";
import { AlertCircle, CheckCircle2, Loader2, MessageCircleQuestion } from "lucide-react";
import type { ButtonHTMLAttributes, HTMLAttributes, Ref } from "react";
import { cn, statusInfo, type Tone } from "../lib/utils";

// Shown on every forecast surface (PRODUCT.md, research use only). Quiet text, never an alert.
export const RESEARCH_USE_NOTICE =
  "Research use only. Forecasts use non-commercial TimesFM 3 weights. Do not use these numbers for commercial decisions or client deliverables.";

export function ResearchNotice({ className }: { className?: string }) {
  return <p className={cn("text-[0.9375rem] text-sand-1", className)}>{RESEARCH_USE_NOTICE}</p>;
}

// One button shape (8 px radius, 40 px high), three weights. Every action in the console uses one of these.
export const buttonVariants = cva(
  "inline-flex min-h-10 shrink-0 items-center justify-center gap-2 rounded-lg px-4 text-base font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-45",
  {
    variants: {
      variant: {
        primary: "bg-ember text-night-0 hover:bg-ember-strong",
        secondary: "border border-sand-2/50 text-sand-0 hover:border-sand-1 hover:bg-night-2",
        ghost: "text-sand-1 hover:bg-night-2 hover:text-sand-0",
      },
      size: { default: "", sm: "min-h-10 px-3 text-[0.9375rem] sm:min-h-9", icon: "size-10 px-0" },
    },
    defaultVariants: { variant: "primary", size: "default" },
  },
);

export function Button({
  className,
  variant,
  size,
  ref,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & VariantProps<typeof buttonVariants> & { ref?: Ref<HTMLButtonElement> }) {
  return <button ref={ref} type="button" className={cn(buttonVariants({ variant, size }), className)} {...props} />;
}

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("rounded-xl border border-sand-2/25 bg-night-1", className)} {...props} />;
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 aria-hidden="true" className={cn("spin size-4", className)} />;
}

const toneClass: Record<Tone, string> = {
  ok: "text-ok",
  bad: "text-bad",
  ember: "text-ember",
  sand: "text-sand-1",
};

// Icon plus word, never colour alone: the state reads the same for everyone.
export function StatusBadge({ status, className }: { status: string; className?: string }) {
  const info = statusInfo(status);
  const Icon =
    status === "created" || status === "running"
      ? null
      : status === "waiting_user"
        ? MessageCircleQuestion
        : info.tone === "ok"
          ? CheckCircle2
          : AlertCircle;
  return (
    <span className={cn("inline-flex items-center gap-1.5 text-[0.9375rem]", toneClass[info.tone], className)}>
      {Icon ? <Icon aria-hidden="true" className="size-4" /> : <Spinner />}
      {info.label}
    </span>
  );
}
