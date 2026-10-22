const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const base = process.env.TEST_BASE_URL || "http://127.0.0.1:8765";
const output = process.env.ARTIFACT_DIR || path.resolve(".test-artifacts");

(async () => {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE }
      : {}),
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1100 },
      reducedMotion: "reduce",
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.setDefaultTimeout(10000);
    await page.goto(base);
    await page.locator(".topic-card").first().waitFor();
    const tree = await (
      await page.request.get(base + "/docs/index.json")
    ).json();
    const articles = [];
    const collect = (node) => {
      if (
        node.type === "file" &&
        node.path.endsWith(".md") &&
        !/\/README\.md$/i.test(node.path)
      )
        articles.push(node);
      (node.children || []).forEach(collect);
    };
    collect(tree);
    const categoryOf = (article) => article.path.split("/")[1];
    const categories = [...new Set(articles.map(categoryOf))];
    assert.equal(await page.locator(".topic-card").count(), categories.length);
    assert.equal(
      await page.locator("#post-list .post-card").count(),
      articles.length,
    );
    for (const category of categories) {
      const expected = articles.filter(
        (article) => categoryOf(article) === category,
      ).length;
      assert.equal(
        Number(
          await page
            .locator(
              `.filter-button[data-category="${category}"] .filter-count`,
            )
            .textContent(),
        ),
        expected,
      );
      assert.ok(
        (
          await page
            .locator(`.topic-card[data-category="${category}"] .topic-number`)
            .textContent()
        ).includes(String(expected).padStart(2, "0")),
      );
    }
    const expectCategory = async (category) => {
      await page.waitForFunction(
        (value) =>
          document.querySelector('.filter-button[aria-pressed="true"]').dataset
            .category === value,
        category,
      );
      const expected =
        category === "全部"
          ? articles.length
          : articles.filter((article) => categoryOf(article) === category)
              .length;
      assert.equal(
        await page.locator("#post-list .post-card").count(),
        expected,
      );
      assert.equal(
        await page.locator(".topic-card.active").count(),
        category === "全部" ? 0 : 1,
      );
    };
    await page.locator('.topic-card[data-category="infra"]').click();
    await expectCategory("infra");
    assert.equal(new URL(page.url()).searchParams.get("topic"), "infra");
    assert.equal(
      await page.locator("#notes-heading").textContent(),
      "推理与系统",
    );
    assert.equal(
      await page.evaluate(() => document.activeElement.id),
      "notes-title",
    );
    await page.locator("#sort-order").selectOption("title");
    await page.reload();
    await expectCategory("infra");
    assert.equal(await page.locator("#sort-order").inputValue(), "title");
    const titles = await page.locator("#post-list h3").allTextContents();
    assert.deepEqual(
      titles,
      [...titles].sort((a, b) => a.localeCompare(b, "zh-CN")),
    );
    await page.locator('.filter-button[data-category="RAG"]').click();
    await expectCategory("RAG");
    await page.goBack();
    await expectCategory("infra");
    await page.goForward();
    await expectCategory("RAG");
    const permalink = await page
      .locator('.topic-card[data-category="数学"]')
      .getAttribute("href");
    const direct = await browser.newPage();
    await direct.goto(new URL(permalink, base).href);
    await direct.waitForSelector(
      '.filter-button[data-category="数学"][aria-pressed="true"]',
    );
    assert.equal(await direct.locator("#post-list .post-card").count(), 1);
    assert.equal(await direct.locator("#sort-order").inputValue(), "title");
    await direct.close();
    await page.locator('.topic-card[data-category="数学"]').focus();
    await page.keyboard.press("Enter");
    await expectCategory("数学");
    await page.locator("#topic-reset").click();
    await expectCategory("全部");
    assert.equal(await page.locator("#sort-order").inputValue(), "title");
    await page.goto(base + "/?topic=missing&sort=invalid#notes");
    await expectCategory("全部");
    assert.equal(await page.locator("#sort-order").inputValue(), "recent");
    for (const width of [320, 390, 768, 1440]) {
      await page.setViewportSize({ width, height: 1100 });
      await page.goto(base + "/#topics");
      await page.locator(".topic-card").first().waitFor();
      assert.ok(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth + 1,
        ),
        `Page overflow at ${width}`,
      );
      const cardsFit = await page
        .locator(".topic-card")
        .evaluateAll((cards) =>
          cards.every((card) => card.scrollWidth <= card.clientWidth + 1),
        );
      assert.ok(cardsFit, `Topic card overflow at ${width}`);
      await page.locator("#topics").scrollIntoViewIfNeeded();
      if ([390, 1440].includes(width))
        await page
          .locator("#topics")
          .screenshot({ path: path.join(output, `topics-${width}.png`) });
      await page.locator('.topic-card[data-category="架构设计"]').click();
      await expectCategory("架构设计");
      await page.locator('.filter-button[data-category="infra"]').click();
      await expectCategory("infra");
      assert.ok(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth + 1,
        ),
        `Filtered overflow at ${width}`,
      );
      if (width === 390)
        await page.screenshot({
          path: path.join(output, "topics-filter-mobile.png"),
        });
    }
    await page.getByRole("button", { name: "切换到深色模式" }).click();
    await page.locator("#topics").scrollIntoViewIfNeeded();
    await page
      .locator("#topics")
      .screenshot({ path: path.join(output, "topics-dark.png") });
    await page.getByRole("link", { name: "书架", exact: true }).click();
    await page.locator("#collection-view").waitFor({ state: "visible" });
    await page.getByRole("link", { name: "文章", exact: true }).click();
    await expectCategory("全部");
    assert.deepEqual(errors, []);
    console.log(
      "PASS: topic counts, shareable filters, sort, reload, history, keyboard, fallback, collections and 320/390/768/1440 layouts",
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
