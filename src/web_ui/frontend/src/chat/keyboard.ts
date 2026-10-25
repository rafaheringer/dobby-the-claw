// Browser bars, zoom and rounding leave small gaps between the layout and visual viewports; a
// keyboard is always far taller. Treating a gap as a keyboard pinned the box under the tab bar.
const MIN_KEYBOARD_PX = 120;

/** How much of the layout viewport the on-screen keyboard covers (0 without a keyboard). */
export function keyboardInset(innerHeight: number, viewportHeight: number, viewportTop: number): number {
  const covered = innerHeight - viewportHeight - viewportTop;
  return covered >= MIN_KEYBOARD_PX ? Math.round(covered) : 0;
}
