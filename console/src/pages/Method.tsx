import { PageHeader, Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui";
import { Example, FlowMerge, FlowNode, FlowRow } from "@/components/Flow";

/** How it works, as four short flows. Each has a plain example, so a non-technical reader can follow the logic. */
export function Method() {
  return (
    <>
      <PageHeader eyebrow="How it works" title="From the business, to a forecast, to a decision"
        description="Four short flows. Each one answers a question a manager would ask, and ends with a worked example." />

      <Card className="mb-6">
        <CardHeader><CardTitle>1 · How does it know what is coming?</CardTitle><CardDescription>It reads the business's own history, up to the morning of the decision. Nothing from the future.</CardDescription></CardHeader>
        <CardContent>
          <FlowRow>
            <FlowNode step={1} title="Read the business" sub="Orders, stock and use-by dates, purchase orders, prices. Read-only." />
            <FlowNode step={2} title="Make weekly series" sub="Units per product, spend per customer, cost per unit. Only complete weeks." />
            <FlowNode step={3} title="TimesFM 3 forecasts" sub="The typical case (median) and a range (10% to 90%) for each series." tone="accent" />
            <FlowNode step={4} title="Compare with simple rules" sub="Moving averages are the bar to beat. If TimesFM is no better, it says so." />
            <FlowNode step={5} title="An agent decides" sub="Works out the expected loss over the whole range, in dollars, and drafts what to do." />
          </FlowRow>
          <Example>
            <p><span className="font-semibold">Seafood P-0120, Friday 3 April.</span> 185 units on hand and 477 inbound, so 662 in total. The forecast says about 2,125 units will sell in the next 5 days (range up to 2,481).</p>
            <p>The agent compares the two and raises: "Likely to run short. Raise a purchase order for about 2,206 units, needed by 8 April." The same case is in Proposals when the lab date is 3 April.</p>
          </Example>
        </CardContent>
      </Card>

      <Card className="mb-6">
        <CardHeader><CardTitle>2 · When does an agent look?</CardTitle><CardDescription>Three ways to wake up. All three go through the same checks, so nobody gets the same alert twice.</CardDescription></CardHeader>
        <CardContent>
          <FlowMerge
            inputs={[
              <FlowNode key="s" title="On a schedule" sub="Every morning before the briefing. Mondays for customer patterns and supplier costs." />,
              <FlowNode key="e" title="When something happens" sub="An order is entered, a purchase order is received or runs late, a price changes." />,
              <FlowNode key="r" title="When someone asks" sub="From the briefing or the forecast explorer." />,
            ]}>
            <FlowNode title="The agent looks" sub="Sees only what was known at that moment." tone="accent" />
          </FlowMerge>
          <div className="mt-3">
            <FlowRow>
              <FlowNode title="Checks first" sub="Same issue already raised? Inside its cooldown? Quiet hours? Has the business taught us to ignore it?" tone="warn" />
              <FlowNode title="Worth a person's time?" sub="Expected loss recovered, less review time and the cost of a false alarm. Each owner sees their best few; the rest wait in a digest." tone="warn" />
              <FlowNode title="Owner is told" sub="One short message with the evidence, in the morning briefing or straight away if it can stop a mistake." />
            </FlowRow>
          </div>
          <Example>
            <p><span className="font-semibold">An order is entered at 2 pm:</span> 800 units of a line a customer normally orders 9 at a time. The check runs the moment the order is saved, not the next morning, so the line can be held before anyone picks it.</p>
            <p><span className="font-semibold">A lot of seafood has 4 days left:</span> that is a morning job. It does not need an instant alert, it needs the sales team to see it with a suggested offer.</p>
          </Example>
        </CardContent>
      </Card>

      <Card className="mb-6">
        <CardHeader><CardTitle>3 · What happens to a proposal?</CardTitle><CardDescription>Nothing leaves the lab. In FoodFlow each outcome would be a draft that a person confirms.</CardDescription></CardHeader>
        <CardContent>
          <FlowRow>
            <FlowNode title="Open" sub="Waiting for its owner, with the evidence attached." tone="accent" />
            <div className="flex min-w-0 flex-1 flex-col gap-2">
              <FlowNode title="Approved" sub="Becomes a draft: a purchase order, an offer, a held order line." />
              <FlowNode title="Adjusted" sub="The owner changes it, then it becomes a draft." />
              <FlowNode title="Dismissed, with a reason" sub={`"One-off", "known promotion", "wrong alert" or "not useful now".`} tone="muted" />
            </div>
          </FlowRow>
          <p className="eyebrow my-3">Only a dismissal teaches the lab</p>
          <FlowRow>
            <FlowNode title="A lesson is created" sub="Stops the same alert for that customer or product for a set period. Visible on the Agents page." />
            <FlowNode title="Undo, any time" sub="Removes the lesson at once. The alert can come back." tone="muted" />
          </FlowRow>
          <Example title="A real example">
            <p>The sales manager dismisses a seafood sell-down with "known promotion". The agent will not raise that product again for 28 days. If it was the wrong call, one click on <span className="font-semibold">Undo</span> restores it.</p>
          </Example>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>4 · What it would look like in production</CardTitle><CardDescription>Proposed, not built. AWS Sydney only, so client data stays in Australia. This lab proves the method before any of it is built.</CardDescription></CardHeader>
        <CardContent>
          <FlowRow>
            <FlowNode title="FoodFlow" sub="Orders, stock, purchases, prices." />
            <FlowNode title="Event feed" sub="Read-only copies of what changed. No writes back." />
            <FlowNode title="Forecast service" sub="A separate service running the model, in batches, behind the Axis model gateway." tone="accent" />
            <FlowNode title="Axis proposals" sub="Evidence, undo, and a person's OK before anything changes." />
          </FlowRow>
          <p className="mt-3 text-xs text-muted-foreground">Gate: TimesFM 3.0 weights are licensed for non-commercial use only. This production shape cannot start until that is resolved (or another model passes the same tests).</p>
        </CardContent>
      </Card>
    </>
  );
}
