import { chromium } from "@playwright/test";
import fs from "node:fs/promises";
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  viewport: { width: 1512, height: 1080 },
  recordVideo: {
    dir: "../docs/recordings",
    size: { width: 1512, height: 1080 },
  },
});
const page = await context.newPage();
await page.goto("http://127.0.0.1:5176");
await page.getByText("API connected").waitFor();
await page.waitForTimeout(2000);
await page.screenshot({ path: "../docs/dashboard.png", fullPage: true });
await page.getByRole("button", { name: "Model registry", exact: true }).click();
await page.waitForTimeout(2200);
await page.screenshot({ path: "../docs/registry.png", fullPage: true });
await page.getByRole("button", { name: "Monitoring", exact: true }).click();
await page.waitForTimeout(2200);
await page.screenshot({ path: "../docs/drift.png", fullPage: true });
await page.getByRole("button", { name: "vibration", exact: true }).click();
await page.waitForTimeout(1500);
await page.getByRole("button", { name: "Inference lab", exact: true }).click();
await page.getByRole("button", { name: "Score reading" }).click();
await page.getByText("Estimated failure probability").waitFor();
await page.waitForTimeout(2200);
await page.screenshot({ path: "../docs/inference.png", fullPage: true });
await context.close();
await fs.rename(await page.video().path(), "../docs/demo.webm");
const mlpage = await browser.newPage({
  viewport: { width: 1440, height: 1000 },
});
await mlpage.goto("http://127.0.0.1:5000/#/experiments");
await mlpage.getByText('Model training',{exact:true}).click();
await mlpage.getByRole('link',{name:/ForgeOps/}).first().click();
await mlpage.getByText('random-forest',{exact:true}).first().waitFor({timeout:30000});
await mlpage.waitForTimeout(2500);
await mlpage.screenshot({ path: "../docs/mlflow.png", fullPage: true });
await browser.close();
console.log("Recorded ForgeOps demo and five screenshots.");
