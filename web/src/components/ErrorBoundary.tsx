import { Component, type ReactNode } from "react";

type Props = { label: string; children: ReactNode };
type State = { failed: boolean };

// Keeps one broken section from blanking the whole Settings dialog. The rest of the page stays usable, and the
// section offers a retry instead of a blank area.
export class ErrorBoundary extends Component<Props, State> {
  state: State = { failed: false };

  static getDerivedStateFromError(): State {
    return { failed: true };
  }

  componentDidCatch(error: unknown) {
    console.error(`${this.props.label} could not be shown`, error);
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div role="alert" className="grid justify-items-start gap-2 rounded-lg border border-bad/60 bg-night-0/50 px-3 py-2.5">
        <p className="text-base text-bad">{this.props.label} could not be shown.</p>
        <button
          type="button"
          onClick={() => this.setState({ failed: false })}
          className="text-caption font-semibold text-sand-0 underline underline-offset-4 focus-visible:outline-2 focus-visible:outline-ember"
        >
          Try again
        </button>
      </div>
    );
  }
}
