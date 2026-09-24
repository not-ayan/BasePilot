# Deploy mechanics

What BasePilot actually does during a raid, for anyone tuning strategies or reading the code.

For the recommended army and the setup that makes it work, see
[Army setup](../README.md#army-setup) in the README.

## Valkyries, Sneaky Goblins, Super Minions

These three share one routine. Per raid, in order:

1. **Select the troop** in the deploy bar. Abort if its icon isn't there.
2. **Drag-deploy around the perimeter.** One continuous press-and-drag from a randomly chosen
   corner, through four corner segments, then release — this empties the selected slot in a line
   along the base edges. Start corner and direction are randomized every raid (at 16:9 it starts
   left or right; at 16:10 it may also start from the top), and each segment's duration is
   jittered ±10%.
3. **Secondary troop**, if enabled on the Run page. A user supplied troop icon image is matched
   against the deploy bar, then the troop is deployed with the configured number of individual
   taps around the perimeter. Use a tight crop of the troop icon at the current game resolution.
4. **Super Dragon**, if it's in the bar.
5. **Siege machine** — Log Launcher first, else Siege Barracks.
6. **Heroes**, in random order each raid: Queen, Warden, Royal Champion, King, Prince, Dragon
   Duke. Whichever are present get dropped, then clicked a second time to fire their abilities.
7. **Earthquake spells.**

Because step 2 is a drag rather than individual taps, exact troop count matters less than
filling the slot — the drag spreads whatever you're carrying along the edges.

## Earthquake placement

BasePilot clicks **11 earthquake points** per raid, in one of two patterns
(*Settings → Earthquake*):

- **Curve placement** (default) — samples an arc through the left/top/right anchor points with
  ±100px of jitter per point.
- **Random placement** — random points inside the region between that arc and a horizontal line
  40% up the frame. Falls back to curve placement if that region comes out degenerate.

It clicks all 11 points regardless of how many Earthquakes you're carrying; once you're out, the
remaining clicks land harmlessly.

## Edrags

A different routine: **12 Electro Dragons** placed individually around the diamond perimeter,
0.2s apart, rather than dragged. The optional secondary troop deploys after those 12 placements.
Upload or paste a tight crop of its deploy-bar icon in the Run page. Heroes and spells follow as above.

## Builder Base

Baby Dragon is the only supported strategy. Night Witches is listed in the UI but is still under
development.
