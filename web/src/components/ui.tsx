import { cva, type VariantProps } from "class-variance-authority";
import { AlertCircle, CheckCircle2, Loader2, MessageCircleQuestion } from "lucide-react";
import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode, Ref } from "react";
import { cn, statusInfo, type Tone } from "../lib/utils";

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
      size: { default: "", sm: "min-h-10 px-3 text-caption sm:min-h-9", icon: "size-10 px-0" },
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

// Icon-only action. The label is both the accessible name and the visible tooltip, so a sighted person and a screen
// reader user get the same words. The tooltip appears on hover and on keyboard focus, never on touch-only devices.
export function IconButton({
  label,
  icon,
  tone = "neutral",
  className,
  disabled,
  ref,
  ...props
}: Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children"> & {
  label: string;
  icon: ReactNode;
  tone?: "neutral" | "danger" | "primary";
  ref?: Ref<HTMLButtonElement>;
}) {
  return (
    <span className="group/tip relative inline-flex shrink-0">
      <button
        ref={ref}
        type="button"
        aria-label={label}
        disabled={disabled}
        className={cn(
          "grid size-10 place-items-center rounded-lg transition-[background-color,color,transform] duration-150 ease-out active:scale-95 disabled:cursor-not-allowed disabled:opacity-45 disabled:active:scale-100 motion-reduce:transition-none motion-reduce:active:scale-100",
          tone === "neutral" && "text-sand-1 hover:bg-night-2 hover:text-sand-0",
          tone === "danger" && "text-sand-1 hover:bg-bad/15 hover:text-bad",
          tone === "primary" && "bg-ember text-night-0 hover:bg-ember-strong",
          className,
        )}
        {...props}
      >
        {icon}
      </button>
      <span
        aria-hidden="true"
        className="pointer-events-none absolute top-full left-1/2 z-50 mt-1.5 -translate-x-1/2 rounded-md border border-sand-2/30 bg-night-3 px-2 py-1 text-micro max-w-60 text-left whitespace-normal text-sand-0 opacity-0 shadow-lg transition-opacity duration-150 group-hover/tip:opacity-100 group-focus-within/tip:opacity-100 motion-reduce:transition-none [@media(hover:none)]:hidden"
      >
        {label}
      </span>
    </span>
  );
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
    <span className={cn("inline-flex items-center gap-1.5 text-caption", toneClass[info.tone], className)}>
      {Icon ? <Icon aria-hidden="true" className="size-4" /> : <Spinner />}
      {info.label}
    </span>
  );
}
