// The search page in a real browser, against a running Space.
//
//   docker run --rm --network host -v "$PWD/tests:/home/pptruser/tests:ro" -v /tmp/shots:/out \
//     -e SPACE=http://localhost:7860 -e SHOTS=/out ghcr.io/puppeteer/puppeteer:23.11.1 \
//     node tests/page_test.mjs
//
// It needs the model up (it waits for /health/model): a suggestion is the
// model's. Screenshots, light and dark, desktop and phone, go to $SHOTS.
import puppeteer from "puppeteer";

const SPACE = process.env.SPACE || "http://localhost:7860";
const SHOTS = process.env.SHOTS || "";
const failures = [];
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failures.push(what); };

for (let i = 0; i < 120; i++) {
  try { const r = await fetch(`${SPACE}/health/model`); if ((await r.json()).model === "up") break; } catch {}
  await new Promise(r => setTimeout(r, 2000));
}

const browser = await puppeteer.launch({ args: ["--no-sandbox"] });
const errors = [];

async function open(width, height, scheme) {
  const page = await browser.newPage();
  await page.setViewport({ width, height });
  await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: scheme }]);
  page.on("pageerror", e => errors.push(String(e)));
  page.on("console", m => { if (m.type() === "error") errors.push(m.text()); });
  await page.goto(SPACE + "/", { waitUntil: "networkidle0" });
  return page;
}
const shot = async (page, name) => { if (SHOTS) await page.screenshot({ path: `${SHOTS}/${name}.png`, fullPage: true }); };
const items = page => page.$$eval("#list li .text", els => els.map(e => e.textContent));
const waitItems = page => page.waitForFunction(() => document.querySelectorAll("#list li").length > 0
                                                   && !document.getElementById("list").hidden, { timeout: 60000 });

const page = await open(1000, 860, "light");
check(await page.$eval("#status", e => e.classList.contains("up")), "the status line says the model is ready");

await page.click("#q");
await page.keyboard.type("maayong ", { delay: 40 });
await waitItems(page);
const next = await items(page);
console.log("     maayong ->", next);
check(next.length > 0 && next.every(t => t.startsWith("maayong ")), "after a space, next words are suggested");
const ghost = await page.$eval("#ghost .rest", e => e.textContent);
check(ghost.length > 0 && next[0] === "maayong " + ghost, "the best one is the grey text");
await shot(page, "desktop-light-next");

await page.keyboard.press("Tab");
check(await page.$eval("#q", e => e.value) === next[0] + " ", "Tab takes the grey text and a space");

// Clear, and type part of a word.
await page.click("#q", { clickCount: 3 });
await page.keyboard.press("Backspace");
await page.keyboard.type("sala", { delay: 40 });
await waitItems(page);
const words = await items(page);
console.log("     sala ->", words);
check(words.every(t => t.startsWith("sala")), "a word is completed from what is typed");
await page.keyboard.press("ArrowDown");
check(await page.$eval("#q", e => e.getAttribute("aria-activedescendant")) === "opt-0", "the arrow keys choose a suggestion");
await shot(page, "desktop-light-word");
await page.keyboard.press("Enter");
check((await page.$eval("#q", e => e.value)).startsWith(words[0]), "Enter on a suggestion takes it");

await page.click('.directions button[data-lang="en"]');
check(await page.$eval('.directions button[data-lang="en"]', e => e.getAttribute("aria-checked")) === "true",
      "a direction can be chosen");

const dark = await open(1000, 860, "dark");
await dark.click("#q");
await dark.keyboard.type("ano ang ", { delay: 40 });
await waitItems(dark);
await shot(dark, "desktop-dark");

const phone = await open(390, 844, "light");
await phone.click("#q");
await phone.keyboard.type("palang", { delay: 40 });
await waitItems(phone).catch(() => {});
const overflow = await phone.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
check(!overflow, "no sideways scrolling on a phone");
await shot(phone, "phone-light");

check(errors.length === 0, `no errors in the console${errors.length ? ": " + errors.join(" | ") : ""}`);
await browser.close();
if (failures.length) { console.log(`\n${failures.length} failed`); process.exit(1); }
console.log("\nall passed");
