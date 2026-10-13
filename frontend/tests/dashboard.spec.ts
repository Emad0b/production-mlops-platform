import { test, expect } from "@playwright/test";

test("measured overview, registry, drift, prediction lineage, and mobile layout", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByText("API connected")).toBeVisible({ timeout: 30000 });
  await expect(page.getByText("CHAMPION PR-AUC")).toBeVisible();
  await expect(page.locator('.recharts-area-curve')).toHaveAttribute('d',/^M/);
  await expect(page.locator('.recharts-rectangle').first()).toBeVisible();
  await page
    .getByRole("button", { name: "Model registry", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Model comparison & promotion" }),
  ).toBeVisible();
  await expect(
    page.getByText("Promotion blocked. The champion remains unchanged.", {
      exact: false,
    }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Monitoring", exact: true }).click();
  await expect(
    page.getByText("Drift detected. Deployment unchanged."),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Evidently report" }),
  ).toHaveAttribute("href", /\/api\/reports\/.+\/html/);
  await page
    .getByRole("button", { name: "Inference lab", exact: true })
    .click();
  await page.getByRole("button", { name: "Score reading" }).click();
  await expect(page.getByText("Estimated failure probability")).toBeVisible();
  await expect(page.getByText("Prediction ID", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "No failure observed" }).click();
  await expect(page.getByRole("alert")).toContainText("administrator token");
  await page.getByRole("button", {name:"Release control", exact:true}).click();
  await expect(page.getByRole("heading", {name:"Retraining proposals"})).toBeVisible();
  await expect(page.getByRole("heading", {name:"Canary deployments"})).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole("button", { name: "Toggle menu" })).toBeVisible();
  await expect
    .poll(() =>
      page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
    )
    .toBe(true);
});
