import * as React from "react";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import * as SwitchPrimitive from "@radix-ui/react-switch";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { AlertTriangle, Loader2, X } from "lucide-react";
import { cn } from "@/lib/utils";

/* Buttons follow FoodFlow's levels: primary (default), secondary (outline), tertiary (ghost), destructive. */
const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors cursor-pointer disabled:cursor-not-allowed disabled:pointer-events-none disabled:opacity-50 active:translate-y-px focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:brightness-95",
        cta: "bg-cta text-cta-foreground hover:brightness-95",
        outline: "border border-input bg-background hover:bg-menu-hover",
        ghost: "hover:bg-menu-hover",
        destructive: "bg-destructive text-white hover:brightness-95",
      },
      size: { default: "h-9 px-4", sm: "h-8 px-3 text-xs", lg: "h-11 px-6 text-base", icon: "h-9 w-9" },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);
export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> { asChild?: boolean; loading?: boolean }
export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, asChild, loading, children, disabled, ...p }, ref) => {
  const Comp = asChild ? Slot : "button";
  return (
    <Comp ref={ref} className={cn(buttonVariants({ variant, size }), className)} disabled={disabled || loading} {...p}>
      {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
      {children}
    </Comp>
  );
});
Button.displayName = "Button";

/* Soft badges: steady fill, no hover change (labels, not controls). */
const badgeVariants = cva("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium whitespace-nowrap", {
  variants: {
    variant: {
      neutral: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
      info: "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300",
      success: "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300",
      warning: "bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300",
      error: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300",
      brand: "bg-accent text-teal dark:text-accent-foreground",
      outline: "border border-input bg-background",
    },
  },
  defaultVariants: { variant: "neutral" },
});
export const Badge = ({ className, variant, ...p }: React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>) => (
  <span className={cn(badgeVariants({ variant }), className)} {...p} />
);

export const Card = ({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("min-w-0 rounded-lg border bg-card text-card-foreground shadow-[0_1px_2px_rgba(19,42,59,0.06)]", className)} {...p} />
);
export const CardHeader = ({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) => <div className={cn("flex flex-col gap-1 p-4 pb-2", className)} {...p} />;
export const CardTitle = ({ className, ...p }: React.HTMLAttributes<HTMLHeadingElement>) => <h3 className={cn("text-base font-semibold leading-tight", className)} {...p} />;
export const CardDescription = ({ className, ...p }: React.HTMLAttributes<HTMLParagraphElement>) => <p className={cn("text-xs text-muted-foreground", className)} {...p} />;
export const CardContent = ({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) => <div className={cn("p-4 pt-2", className)} {...p} />;

export const Skeleton = ({ className }: { className?: string }) => <div className={cn("skeleton h-4 w-full", className)} aria-hidden />;

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(({ className, ...p }, ref) => (
  <input ref={ref} className={cn("h-9 w-full rounded-md border border-input bg-background px-3 text-sm placeholder:text-muted-foreground focus-visible:outline-2 focus-visible:outline-ring", className)} {...p} />
));
Input.displayName = "Input";

export const Select = ({ className, ...p }: React.SelectHTMLAttributes<HTMLSelectElement>) => (
  <select className={cn("h-9 rounded-md border border-input bg-background px-2 text-sm focus-visible:outline-2 focus-visible:outline-ring", className)} {...p} />
);

export const Switch = ({ className, ...p }: React.ComponentProps<typeof SwitchPrimitive.Root>) => (
  <SwitchPrimitive.Root className={cn("peer inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full border-2 border-transparent bg-slate-300 transition-colors data-[state=checked]:bg-primary disabled:cursor-not-allowed disabled:opacity-50 dark:bg-slate-600", className)} {...p}>
    <SwitchPrimitive.Thumb className="pointer-events-none block h-4 w-4 rounded-full bg-white shadow transition-transform data-[state=checked]:translate-x-4" />
  </SwitchPrimitive.Root>
);

export const Tabs = TabsPrimitive.Root;
export const TabsList = ({ className, ...p }: React.ComponentProps<typeof TabsPrimitive.List>) => (
  <TabsPrimitive.List className={cn("inline-flex h-9 items-center gap-1 rounded-lg bg-[#e7ecf2] p-1 dark:bg-slate-800", className)} {...p} />
);
export const TabsTrigger = ({ className, ...p }: React.ComponentProps<typeof TabsPrimitive.Trigger>) => (
  <TabsPrimitive.Trigger className={cn("inline-flex h-7 items-center rounded-md px-3 text-sm font-medium text-muted-foreground transition-colors cursor-pointer data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-sm", className)} {...p} />
);
export const TabsContent = TabsPrimitive.Content;

/* Sheet: supplementary detail, closes on outside click and Escape (FoodFlow overlay rule). */
export const Sheet = DialogPrimitive.Root;
export function SheetContent({ className, children, title, description, ...p }: React.ComponentProps<typeof DialogPrimitive.Content> & { title: string; description?: string }) {
  return (
    <DialogPrimitive.Portal>
      <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/40" />
      <DialogPrimitive.Content className={cn("fixed right-0 top-0 z-50 flex h-full w-full max-w-2xl flex-col bg-background shadow-[var(--menu-shadow)] border-l focus:outline-none", className)} {...p}>
        <div className="flex items-start justify-between gap-4 border-b p-4">
          <div className="min-w-0">
            <DialogPrimitive.Title className="text-base font-semibold leading-snug">{title}</DialogPrimitive.Title>
            {description && <DialogPrimitive.Description className="mt-0.5 text-xs text-muted-foreground">{description}</DialogPrimitive.Description>}
          </div>
          <DialogPrimitive.Close asChild><Button variant="ghost" size="icon" aria-label="Close"><X className="h-4 w-4" /></Button></DialogPrimitive.Close>
        </div>
        <div className="flex-1 overflow-y-auto">{children}</div>
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  );
}

export function InlineAlert({ tone = "error", title, children, onRetry }: { tone?: "error" | "warning" | "info"; title: string; children?: React.ReactNode; onRetry?: () => void }) {
  const t = { error: "border-red-300 bg-red-50 text-red-900 dark:bg-red-950 dark:text-red-100", warning: "border-amber-300 bg-amber-50 text-amber-900 dark:bg-amber-950 dark:text-amber-100", info: "border-blue-300 bg-blue-50 text-blue-900 dark:bg-blue-950 dark:text-blue-100" }[tone];
  return (
    <div role="alert" className={cn("flex items-start gap-3 rounded-lg border p-3 text-sm", t)}>
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <div className="min-w-0 flex-1"><p className="font-semibold">{title}</p>{children && <div className="mt-0.5 text-xs opacity-90">{children}</div>}</div>
      {onRetry && <Button size="sm" variant="outline" onClick={onRetry}>Retry</Button>}
    </div>
  );
}

export function PageHeader({ eyebrow, title, description, actions }: { eyebrow?: string; title: string; description?: string; actions?: React.ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0 max-w-3xl">
        {eyebrow && <p className="eyebrow mb-1">{eyebrow}</p>}
        <h1 className="text-2xl font-semibold leading-tight tracking-tight">{title}</h1>
        {description && <p className="mt-1 text-sm text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

export function Stat({ label, value, sub, tone = "default", big = false }: { label: string; value: React.ReactNode; sub?: React.ReactNode; tone?: "default" | "good" | "warn" | "bad"; big?: boolean }) {
  const c = { default: "", good: "text-[var(--status-success)]", warn: "text-[var(--status-warning)]", bad: "text-[var(--status-error)]" }[tone];
  return (
    <div className="min-w-0">
      <p className="eyebrow">{label}</p>
      <p className={cn("mt-1 font-semibold tabular leading-none", big ? "text-4xl" : "text-2xl", c)}>{value}</p>
      {sub && <p className="mt-1.5 text-xs text-muted-foreground">{sub}</p>}
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: React.ReactNode }) {
  return <div className="rounded-lg border border-dashed p-8 text-center"><p className="font-medium">{title}</p>{children && <p className="mx-auto mt-1 max-w-md text-xs text-muted-foreground">{children}</p>}</div>;
}

export const SEVERITY: Record<string, { label: string; variant: "error" | "warning" | "neutral" }> = {
  high: { label: "High", variant: "error" }, medium: { label: "Medium", variant: "warning" }, low: { label: "Low", variant: "neutral" },
};
