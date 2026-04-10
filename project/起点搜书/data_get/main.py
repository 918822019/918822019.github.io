import asyncio
import aiohttp
from bs4 import BeautifulSoup
import time
import random
import logging

# 新增：引入playwright
from playwright.async_api import async_playwright

# 配置日志，写入文件和控制台
log_file = "data_get.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


from urllib.parse import urljoin, urlparse


# ================= 多级异步爬虫框架 =================


class MultiLevelAsyncScraper:
    """
    多级异步爬虫框架
    支持多级递归抓取，每一级可自定义解析和新URL处理逻辑。
    """

    async def fetch_with_playwright(self, url, parse_func, wait_selector: str = None):
        """
        使用Playwright无头浏览器抓取页面源码，伪装UA。
        """
        async with self.semaphore:
            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    ua = random.choice(self.user_agents)
                    async with async_playwright() as p:
                        browser = await p.chromium.launch(headless=True)
                        context = await browser.new_context(user_agent=ua)
                        page = await context.new_page()
                        # 等待网络空闲，增大超时，提升动态内容渲染成功率
                        await page.goto(url, timeout=60000, wait_until="networkidle")
                        # 可自定义等待条件，如等待某元素出现
                        # 如果提供了 wait_selector，优先等待该元素出现
                        if wait_selector:
                            try:
                                await page.wait_for_selector(
                                    wait_selector, timeout=30000
                                )
                            except Exception:
                                logger.warning(
                                    f"{url} 未能在30s内出现选择器 {wait_selector}"
                                )
                        # 自动向下滚动几次以触发懒加载内容
                        try:
                            for _ in range(5):
                                await page.evaluate(
                                    "window.scrollBy(0, document.body.scrollHeight)"
                                )
                                await page.wait_for_timeout(800)
                        except Exception:
                            pass
                        await page.wait_for_timeout(1200)  # 稍作等待，确保渲染
                        html = await page.content()
                        await context.close()
                        await browser.close()
                    file_id = self._get_file_id_from_url(url)
                    html_filename = f"page_{file_id}.html"
                    try:
                        with open(html_filename, "w", encoding="utf-8") as f:
                            f.write(html)
                        logger.info(f"已保存HTML源码: {html_filename}")
                    except Exception as e:
                        logger.error(
                            f"保存HTML源码失败: {html_filename}, 错误: {str(e)}"
                        )
                    result = parse_func(url, html)
                    result["html_file"] = html_filename
                    return result
                except Exception as e:
                    wait_time = (2**retry_count) + random.random()
                    logger.error(f"[Playwright]{url} 请求异常: {str(e)}，重试中...")
                    await asyncio.sleep(wait_time)
                    retry_count += 1
            logger.error(f"[Playwright]{url} 最终失败，已达到最大重试次数。")
            return {"url": url, "status": "Failed"}

    def __init__(self, max_concurrent=10, max_retries=3):
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        ]

    async def fetch(self, session, url, parse_func):
        """
        抓取单个页面，并用parse_func解析
        :param session: aiohttp.ClientSession
        :param url: 目标URL
        :param parse_func: 解析函数，返回dict，必须包含'url'和'status'，可选'new_urls'
        """
        async with self.semaphore:
            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    headers = {"User-Agent": random.choice(self.user_agents)}
                    async with session.get(
                        url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status in [429, 500, 502, 503, 504]:
                            wait_time = (2**retry_count) + random.random()
                            logger.warning(
                                f"{url} 状态码 {response.status}，等待 {wait_time:.2f}s 后重试..."
                            )
                            await asyncio.sleep(wait_time)
                            retry_count += 1
                            continue
                        html = await response.text()
                        # 保存html源码到本地文件
                        file_id = self._get_file_id_from_url(url)
                        html_filename = f"page_{file_id}.html"
                        try:
                            with open(html_filename, "w", encoding="utf-8") as f:
                                f.write(html)
                            logger.info(f"已保存HTML源码: {html_filename}")
                        except Exception as e:
                            logger.error(
                                f"保存HTML源码失败: {html_filename}, 错误: {str(e)}"
                            )
                        result = parse_func(url, html)
                        result["html_file"] = html_filename
                        return result
                except Exception as e:
                    wait_time = (2**retry_count) + random.random()
                    logger.error(f"{url} 请求异常: {str(e)}，重试中...")
                    await asyncio.sleep(wait_time)
                    retry_count += 1
            logger.error(f"{url} 最终失败，已达到最大重试次数。")
            return {"url": url, "status": "Failed"}

    def _get_file_id_from_url(self, url):
        """
        根据url提取唯一编号（如/book/1/ -> 1），用于命名html文件
        """
        import re

        # 尝试匹配最后的数字
        match = re.search(r"(\d+)(?=/?$)", url)
        if match:
            return match.group(1)
        # 匹配不到则用hash
        return str(abs(hash(url)))

    async def crawl_one_level(
        self, urls, parse_func, filter_func=None, max_new_urls=20
    ):
        """
        只抓取一级页面，提取子链接，不递归
        :param urls: 待抓取URL列表
        :param parse_func: 解析函数，返回dict，必须包含'url'和'new_urls'
        :param filter_func: 可选，对new_urls做过滤
        :param max_new_urls: 限制新url数量
        :return: (results, all_new_urls)
        """
        logger.info(f"待抓取URL数: {len(urls)} (Playwright)")
        tasks = [self.fetch_with_playwright(url, parse_func) for url in urls]
        results = await asyncio.gather(*tasks)
        logger.info("\n抓取结果:")
        for r in results:
            logger.info(
                f"URL: {r.get('url')} -> 状态: {r.get('status')} 标题: {r.get('title','')} HTML: {r.get('html_file','')}"
            )
        all_new_urls = []
        for r in results:
            if r.get("new_urls"):
                all_new_urls.extend(r["new_urls"])
        if filter_func:
            all_new_urls = filter_func(all_new_urls)
        all_new_urls = list(set(all_new_urls))[:max_new_urls]
        logger.info("\n子链接示例:")
        for u in all_new_urls[:10]:
            logger.info(u)
        print(f"已写入日志文件: {log_file}")
        return results, all_new_urls


# ========== 示例解析函数 ==========
def parse_list_page(url, html):
    """
    示例：解析列表页，提取所有a标签的href为新url
    :param url: 当前页面url
    :param html: 页面html源码
    :return: dict，必须包含'url'、'title'、'status'，可选'new_urls'
    """
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title else "No Title"
    links = set()
    for a in soup.find_all("a", href=True):
        full_url = urljoin(url, a["href"])
        links.add(full_url)
    return {
        "url": url,
        "title": title,
        "status": "Success",
        "new_urls": list(links),
    }


def parse_detail_page(url, html):
    """
    示例：解析详情页，只提取标题
    """
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title else "No Title"
    return {"url": url, "title": title, "status": "Success", "new_urls": []}


def filter_httpbin(urls):
    """
    示例：只保留httpbin.org域名的url
    """
    return [u for u in urls if urlparse(u).netloc == "httpbin.org"]


# ========== 工具函数 ==========


def build_urls(base_url, start, end, pattern=None):
    """
    批量生成递增url
    :param base_url: 基础url，如 'https://xx.com/page/'
    :param start: 起始数字（包含）
    :param end: 结束数字（包含）
    :param pattern: 可选，若有特殊格式如 'https://xx.com/page_{}.html'
    :return: url列表
    """
    if pattern:
        return [pattern.format(i) for i in range(start, end + 1)]
    else:
        return [f"{base_url}{i}" for i in range(start, end + 1)]


def build_anchor_urls(base_url, anchor_prefix, start, end):
    """
    批量生成带锚点的url，如 https://xx.com/#/book/1/
    :param base_url: 基础url（不带#），如 'https://7e0c.bqg504.cc'
    :param anchor_prefix: 锚点前缀，如 '/#/book/'
    :param start: 起始数字
    :param end: 结束数字
    :return: url列表
    """
    return [f"{base_url}{anchor_prefix}{i}/" for i in range(start, end + 1)]


# ========== 主入口 ==========
async def main():
    """
    主入口：配置多级爬虫流程，运行爬虫
    """
    # 示例：批量生成锚点型URL
    anchor_urls = build_anchor_urls("https://7e0c.bqg504.cc", "/#/book/", 1, 3)
    logger.info("示例-锚点型URL:")
    for u in anchor_urls:
        logger.info(u)

    scraper = MultiLevelAsyncScraper(max_concurrent=5, max_retries=2)
    start_time = time.time()
    await scraper.crawl_one_level(
        urls=anchor_urls,
        parse_func=parse_list_page,
        filter_func=None,  # 可自定义过滤
        max_new_urls=20,
    )
    end_time = time.time()
    logger.info(f"\n全部完成，总耗时: {end_time - start_time:.2f} 秒")
    print(f"全部完成，详细内容见 {log_file}")


if __name__ == "__main__":
    # 兼容Windows事件循环策略
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
