from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import random
import sqlite3
import threading
import time
from datetime import datetime
from typing import Any, Iterator
from urllib import error, parse, request


BASE_SITE_URL = "https://7e0c.bqg504.cc"
BOOK_HOME_URL = BASE_SITE_URL + "/#/book/{book_id}/"
CHAPTER_PAGE_URL = BASE_SITE_URL + "/book/{book_id}/{chapter_id}.html"
BOOK_API_URL = "https://apibi.cc/api/book?id={book_id}"
BOOKLIST_API_URL = "https://apibi.cc/api/booklist?id={book_id}"
CHAPTER_API_URL = "https://apibi.cc/api/chapter?id={book_id}&chapterid={chapter_id}"

USER_AGENT_POOL = [
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36 Edg/124.0.2478.67"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.4 Safari/605.1.15"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
]

ACCEPT_LANGUAGE_POOL = [
    "zh-CN,zh;q=0.9,en;q=0.8",
    "zh-CN,zh;q=0.95,en-US;q=0.85,en;q=0.75",
    "zh-CN,zh;q=0.9,en;q=0.7,ja;q=0.4",
]

BASE_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": ACCEPT_LANGUAGE_POOL[0],
    "Referer": BASE_SITE_URL + "/",
    "Connection": "keep-alive",
    "Origin": BASE_SITE_URL,
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "books.db")
LOG_FILE = os.path.join(BASE_DIR, "data_get.log")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_inline_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\u3000", " ").split())


def normalize_multiline_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def chunked(items: list[Any], size: int) -> Iterator[list[Any]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def validate_range(start: int, end: int) -> None:
    if start < 1 or end < 1:
        raise ValueError("start 和 end 必须大于等于 1")
    if end < start:
        raise ValueError("end 不能小于 start")


def make_book_home_url(book_id: int) -> str:
    return BOOK_HOME_URL.format(book_id=book_id)


def make_chapter_page_url(book_id: int, chapter_id: int) -> str:
    return CHAPTER_PAGE_URL.format(book_id=book_id, chapter_id=chapter_id)


def make_book_api_url(book_id: int) -> str:
    return BOOK_API_URL.format(book_id=book_id)


def make_booklist_api_url(book_id: int) -> str:
    return BOOKLIST_API_URL.format(book_id=book_id)


def make_chapter_api_url(book_id: int, chapter_id: int) -> str:
    return CHAPTER_API_URL.format(book_id=book_id, chapter_id=chapter_id)


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def maybe_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def build_referer(url: str) -> str:
    parsed = parse.urlparse(url)
    query = parse.parse_qs(parsed.query)
    book_id = to_int((query.get("id") or ["0"])[0])
    chapter_id = to_int((query.get("chapterid") or ["0"])[0])

    if book_id <= 0:
        return BASE_SITE_URL + "/"
    if chapter_id > 0:
        return make_chapter_page_url(book_id, chapter_id)
    return make_book_home_url(book_id)


def build_request_headers(url: str) -> dict[str, str]:
    headers = dict(BASE_HEADERS)
    headers["User-Agent"] = random.choice(USER_AGENT_POOL)
    headers["Accept-Language"] = random.choice(ACCEPT_LANGUAGE_POOL)
    headers["Referer"] = build_referer(url)
    return headers


class RequestPacer:
    def __init__(self, min_interval: float, jitter: float) -> None:
        self.min_interval = max(0.0, min_interval)
        self.jitter = max(0.0, jitter)
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0

    def wait_for_turn(self) -> None:
        if self.min_interval <= 0 and self.jitter <= 0:
            return

        interval = self.min_interval
        if self.jitter > 0:
            interval += random.uniform(0.0, self.jitter)

        sleep_seconds = 0.0
        with self._lock:
            now = time.monotonic()
            if self._next_allowed_at > now:
                sleep_seconds = self._next_allowed_at - now
                self._next_allowed_at += interval
            else:
                self._next_allowed_at = now + interval

        maybe_sleep(sleep_seconds)


class BookApiClient:
    def __init__(
        self,
        timeout: int,
        retries: int,
        min_request_interval: float,
        request_jitter: float,
        retry_backoff_base: float,
        retry_backoff_max: float,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.min_request_interval = max(0.0, min_request_interval)
        self.request_jitter = max(0.0, request_jitter)
        self.retry_backoff_base = max(0.1, retry_backoff_base)
        self.retry_backoff_max = max(
            self.retry_backoff_base,
            retry_backoff_max,
        )
        self.request_pacer = RequestPacer(
            min_interval=self.min_request_interval,
            jitter=self.request_jitter,
        )

    def build_retry_delay(self, attempt: int, exc: Exception) -> float:
        delay = min(
            self.retry_backoff_max,
            self.retry_backoff_base * (2 ** max(0, attempt - 1)),
        )
        delay += random.uniform(0.0, max(0.2, self.request_jitter))

        if isinstance(exc, error.HTTPError) and exc.code in {429, 522, 523, 524}:
            delay = max(delay, 4.0 + random.uniform(0.5, 1.5))
        return delay

    def fetch_json(self, url: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                self.request_pacer.wait_for_turn()
                req = request.Request(url, headers=build_request_headers(url))
                with request.urlopen(req, timeout=self.timeout) as response:
                    payload = response.read().decode("utf-8", errors="ignore")
                data = json.loads(payload)
                if not isinstance(data, dict):
                    raise RuntimeError(f"接口返回不是对象: {url}")
                return data
            except (
                error.HTTPError,
                error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                RuntimeError,
            ) as exc:
                last_error = exc
                if attempt == self.retries:
                    break
                wait_seconds = self.build_retry_delay(attempt, exc)
                logger.warning(
                    "请求失败，稍后重试: url=%s attempt=%s/%s error=%s",
                    url,
                    attempt,
                    self.retries,
                    exc,
                )
                maybe_sleep(wait_seconds)
        raise RuntimeError(f"请求失败: {url} -> {last_error}")

    def fetch_book_bundle(
        self, book_id: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        book_url = make_book_api_url(book_id)
        catalog_url = make_booklist_api_url(book_id)
        book_payload = self.fetch_json(book_url)
        catalog_payload = self.fetch_json(catalog_url)

        title = normalize_inline_text(book_payload.get("title"))
        chapter_names_raw = catalog_payload.get("list") or []
        if not isinstance(chapter_names_raw, list):
            raise RuntimeError(f"章节目录格式异常: book_id={book_id}")
        chapter_names = [
            normalize_inline_text(name)
            for name in chapter_names_raw
            if normalize_inline_text(name)
        ]
        if not title:
            raise RuntimeError(f"书籍标题为空: book_id={book_id}")

        last_chapter_id = to_int(book_payload.get("lastchapterid"))
        chapter_count = len(chapter_names) or last_chapter_id
        fetched_at = now_iso()

        book_record = {
            "book_id": book_id,
            "title": title,
            "category": normalize_inline_text(book_payload.get("sortname")),
            "author": normalize_inline_text(book_payload.get("author")),
            "serial_status": normalize_inline_text(book_payload.get("full")),
            "intro": normalize_inline_text(book_payload.get("intro")),
            "last_chapter_id": last_chapter_id,
            "last_chapter_title": normalize_inline_text(
                book_payload.get("lastchapter")
            ),
            "last_update": normalize_inline_text(book_payload.get("lastupdate")),
            "dir_id": normalize_inline_text(book_payload.get("dirid")),
            "chapter_count": chapter_count,
            "homepage_url": make_book_home_url(book_id),
            "source_book_api": book_url,
            "source_catalog_api": catalog_url,
            "catalog_fetched_at": fetched_at,
        }

        chapter_records = []
        for chapter_id, chapter_name in enumerate(chapter_names, start=1):
            chapter_records.append(
                {
                    "book_id": book_id,
                    "chapter_id": chapter_id,
                    "chapter_name": chapter_name,
                    "chapter_url": make_chapter_page_url(book_id, chapter_id),
                    "source_api_url": make_chapter_api_url(
                        book_id,
                        chapter_id,
                    ),
                }
            )

        return book_record, chapter_records

    def fetch_chapter_content(
        self,
        book_id: int,
        chapter_id: int,
    ) -> dict[str, Any]:
        api_url = make_chapter_api_url(book_id, chapter_id)
        payload = self.fetch_json(api_url)
        content = normalize_multiline_text(payload.get("txt"))
        chapter_name = normalize_inline_text(payload.get("chaptername"))

        if not chapter_name and not content:
            raise RuntimeError(
                f"章节内容为空: book_id={book_id} chapter_id={chapter_id}"
            )

        return {
            "book_id": book_id,
            "chapter_id": chapter_id,
            "chapter_name": chapter_name,
            "content": content,
            "content_length": len(content),
            "fetched_at": now_iso(),
            "source_api_url": api_url,
        }


def open_database(db_path: str, synchronous_mode: str) -> sqlite3.Connection:
    ensure_parent_dir(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA synchronous={synchronous_mode}")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA wal_autocheckpoint=1000")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            category TEXT,
            author TEXT,
            serial_status TEXT,
            intro TEXT,
            last_chapter_id INTEGER,
            last_chapter_title TEXT,
            last_update TEXT,
            dir_id TEXT,
            chapter_count INTEGER NOT NULL DEFAULT 0,
            homepage_url TEXT NOT NULL,
            source_book_api TEXT NOT NULL,
            source_catalog_api TEXT NOT NULL,
            catalog_fetched_at TEXT,
            content_fetched_chapters INTEGER NOT NULL DEFAULT 0,
            content_completed INTEGER NOT NULL DEFAULT 0,
            last_error TEXT
        );

        CREATE TABLE IF NOT EXISTS chapters (
            book_id INTEGER NOT NULL,
            chapter_id INTEGER NOT NULL,
            chapter_name TEXT NOT NULL,
            chapter_url TEXT NOT NULL,
            source_api_url TEXT NOT NULL,
            content TEXT,
            content_length INTEGER NOT NULL DEFAULT 0,
            is_content_fetched INTEGER NOT NULL DEFAULT 0,
            fetched_at TEXT,
            last_error TEXT,
            PRIMARY KEY (book_id, chapter_id),
            FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_chapters_pending
        ON chapters(is_content_fetched, book_id, chapter_id);
        """
    )
    conn.commit()


def get_existing_catalog_ids(
    conn: sqlite3.Connection, start: int, end: int
) -> set[int]:
    rows = conn.execute(
        """
        SELECT book_id
        FROM books
        WHERE book_id BETWEEN ? AND ?
          AND chapter_count > 0
        """,
        (start, end),
    ).fetchall()
    return {int(row["book_id"]) for row in rows}


def upsert_book_catalog(
    conn: sqlite3.Connection,
    book_record: dict[str, Any],
    chapter_records: list[dict[str, Any]],
) -> None:
    with conn:
        conn.execute(
            """
            INSERT INTO books (
                book_id,
                title,
                category,
                author,
                serial_status,
                intro,
                last_chapter_id,
                last_chapter_title,
                last_update,
                dir_id,
                chapter_count,
                homepage_url,
                source_book_api,
                source_catalog_api,
                catalog_fetched_at,
                last_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            ON CONFLICT(book_id) DO UPDATE SET
                title = excluded.title,
                category = excluded.category,
                author = excluded.author,
                serial_status = excluded.serial_status,
                intro = excluded.intro,
                last_chapter_id = excluded.last_chapter_id,
                last_chapter_title = excluded.last_chapter_title,
                last_update = excluded.last_update,
                dir_id = excluded.dir_id,
                chapter_count = excluded.chapter_count,
                homepage_url = excluded.homepage_url,
                source_book_api = excluded.source_book_api,
                source_catalog_api = excluded.source_catalog_api,
                catalog_fetched_at = excluded.catalog_fetched_at,
                last_error = NULL
            """,
            (
                book_record["book_id"],
                book_record["title"],
                book_record["category"],
                book_record["author"],
                book_record["serial_status"],
                book_record["intro"],
                book_record["last_chapter_id"],
                book_record["last_chapter_title"],
                book_record["last_update"],
                book_record["dir_id"],
                book_record["chapter_count"],
                book_record["homepage_url"],
                book_record["source_book_api"],
                book_record["source_catalog_api"],
                book_record["catalog_fetched_at"],
            ),
        )

        if chapter_records:
            conn.executemany(
                """
                INSERT INTO chapters (
                    book_id,
                    chapter_id,
                    chapter_name,
                    chapter_url,
                    source_api_url
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(book_id, chapter_id) DO UPDATE SET
                    chapter_name = excluded.chapter_name,
                    chapter_url = excluded.chapter_url,
                    source_api_url = excluded.source_api_url
                """,
                [
                    (
                        chapter["book_id"],
                        chapter["chapter_id"],
                        chapter["chapter_name"],
                        chapter["chapter_url"],
                        chapter["source_api_url"],
                    )
                    for chapter in chapter_records
                ],
            )


def record_catalog_error(
    conn: sqlite3.Connection, book_id: int, error_message: str
) -> None:
    with conn:
        conn.execute(
            """
            INSERT INTO books (
                book_id,
                title,
                category,
                author,
                serial_status,
                intro,
                last_chapter_id,
                last_chapter_title,
                last_update,
                dir_id,
                chapter_count,
                homepage_url,
                source_book_api,
                source_catalog_api,
                catalog_fetched_at,
                last_error
            ) VALUES (?, '', '', '', '', '', 0, '', '', '', 0, ?, ?, ?, ?, ?)
            ON CONFLICT(book_id) DO UPDATE SET
                last_error = excluded.last_error,
                catalog_fetched_at = excluded.catalog_fetched_at
            """,
            (
                book_id,
                make_book_home_url(book_id),
                make_book_api_url(book_id),
                make_booklist_api_url(book_id),
                now_iso(),
                error_message,
            ),
        )


def get_books_in_range(
    conn: sqlite3.Connection, start: int, end: int
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
            book_id,
            title,
            chapter_count,
            content_fetched_chapters,
            content_completed
        FROM books
        WHERE book_id BETWEEN ? AND ?
        ORDER BY book_id
        """,
        (start, end),
    ).fetchall()


def get_pending_chapters(
    conn: sqlite3.Connection, book_id: int, limit: int = 0
) -> list[sqlite3.Row]:
    sql = """
        SELECT book_id, chapter_id, chapter_name, source_api_url
        FROM chapters
        WHERE book_id = ?
          AND is_content_fetched = 0
        ORDER BY chapter_id
    """
    params: tuple[int, ...] | tuple[int, int]
    params = (book_id,)
    if limit > 0:
        sql += "\n        LIMIT ?"
        params = (book_id, limit)
    return conn.execute(sql, params).fetchall()


def count_pending_chapters_in_range(
    conn: sqlite3.Connection, start: int, end: int
) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS pending
        FROM chapters
        WHERE book_id BETWEEN ? AND ?
          AND is_content_fetched = 0
        """,
        (start, end),
    ).fetchone()
    return int(row["pending"])


def upsert_chapter_content(
    conn: sqlite3.Connection, chapter_record: dict[str, Any]
) -> None:
    with conn:
        conn.execute(
            """
            INSERT INTO chapters (
                book_id,
                chapter_id,
                chapter_name,
                chapter_url,
                source_api_url,
                content,
                content_length,
                is_content_fetched,
                fetched_at,
                last_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, NULL)
            ON CONFLICT(book_id, chapter_id) DO UPDATE SET
                chapter_name = excluded.chapter_name,
                source_api_url = excluded.source_api_url,
                content = excluded.content,
                content_length = excluded.content_length,
                is_content_fetched = 1,
                fetched_at = excluded.fetched_at,
                last_error = NULL
            """,
            (
                chapter_record["book_id"],
                chapter_record["chapter_id"],
                chapter_record["chapter_name"],
                make_chapter_page_url(
                    chapter_record["book_id"], chapter_record["chapter_id"]
                ),
                chapter_record["source_api_url"],
                chapter_record["content"],
                chapter_record["content_length"],
                chapter_record["fetched_at"],
            ),
        )


def record_chapter_error(
    conn: sqlite3.Connection,
    book_id: int,
    chapter_id: int,
    error_message: str,
) -> None:
    with conn:
        conn.execute(
            """
            UPDATE chapters
            SET last_error = ?
            WHERE book_id = ? AND chapter_id = ?
            """,
            (error_message, book_id, chapter_id),
        )


def refresh_book_content_progress(
    conn: sqlite3.Connection,
    book_id: int,
) -> None:
    row = conn.execute(
        """
        SELECT COUNT(*) AS total,
               COALESCE(SUM(is_content_fetched), 0) AS fetched
        FROM chapters
        WHERE book_id = ?
        """,
        (book_id,),
    ).fetchone()
    total = int(row["total"])
    fetched = int(row["fetched"])
    with conn:
        conn.execute(
            """
            UPDATE books
            SET content_fetched_chapters = ?,
                content_completed = ?,
                last_error = CASE
                    WHEN last_error = '' THEN NULL
                    ELSE last_error
                END
            WHERE book_id = ?
            """,
            (fetched, 1 if total > 0 and fetched == total else 0, book_id),
        )


def crawl_books_stage(
    conn: sqlite3.Connection,
    client: BookApiClient,
    start: int,
    end: int,
    concurrency: int,
    progress_every: int,
    refresh_existing: bool,
) -> None:
    existing_ids = set()
    if not refresh_existing:
        existing_ids = get_existing_catalog_ids(conn, start, end)

    targets = [
        book_id
        for book_id in range(start, end + 1)
        if refresh_existing or book_id not in existing_ids
    ]
    if not targets:
        logger.info("目录阶段无需执行，目标区间已全部存在。")
        return

    total = len(targets)
    success_count = 0
    failure_count = 0
    stage_start = time.time()
    logger.info(
        "目录阶段启动: range=%s-%s targets=%s concurrency=%s",
        start,
        end,
        total,
        concurrency,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_book = {
            executor.submit(client.fetch_book_bundle, book_id): book_id
            for book_id in targets
        }
        for index, future in enumerate(
            concurrent.futures.as_completed(future_to_book), start=1
        ):
            book_id = future_to_book[future]
            try:
                book_record, chapter_records = future.result()
                upsert_book_catalog(conn, book_record, chapter_records)
                success_count += 1
            except Exception as exc:
                failure_count += 1
                record_catalog_error(conn, book_id, str(exc))
                logger.warning("目录抓取失败: book_id=%s error=%s", book_id, exc)

            if index % progress_every == 0 or index == total:
                logger.info(
                    "目录进度: %s/%s success=%s failure=%s elapsed=%.2fs",
                    index,
                    total,
                    success_count,
                    failure_count,
                    time.time() - stage_start,
                )


def crawl_content_stage(
    conn: sqlite3.Connection,
    client: BookApiClient,
    start: int,
    end: int,
    concurrency: int,
    progress_every: int,
    batch_size: int,
    chapter_progress_every: int,
    max_pending_per_book: int,
) -> None:
    books = get_books_in_range(conn, start, end)
    if not books:
        logger.info("正文阶段无可处理书籍，请先执行 crawl-books。")
        return

    pending_chapter_total = count_pending_chapters_in_range(conn, start, end)
    if max_pending_per_book > 0:
        pending_chapter_total = 0
        for book_row in books:
            pending_chapter_total += len(
                get_pending_chapters(
                    conn,
                    int(book_row["book_id"]),
                    max_pending_per_book,
                )
            )

    if pending_chapter_total == 0:
        logger.info("正文阶段无需执行，目标区间章节内容已全部完成。")
        return

    stage_start = time.time()
    completed_books = 0
    chapter_success_count = 0
    chapter_failure_count = 0
    chapter_processed_count = 0
    logger.info(
        (
            "正文阶段启动: range=%s-%s books=%s pending_chapters=%s "
            "concurrency=%s batch_size=%s interval=%.3fs jitter=%.3fs"
        ),
        start,
        end,
        len(books),
        pending_chapter_total,
        concurrency,
        batch_size,
        client.min_request_interval,
        client.request_jitter,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        for book_index, book_row in enumerate(books, start=1):
            book_id = int(book_row["book_id"])
            pending_rows = get_pending_chapters(
                conn,
                book_id,
                max_pending_per_book,
            )
            if not pending_rows:
                refresh_book_content_progress(conn, book_id)
                completed_books += 1
                if book_index % progress_every == 0 or book_index == len(books):
                    logger.info(
                        "正文书籍进度: %s/%s elapsed=%.2fs",
                        completed_books,
                        len(books),
                        time.time() - stage_start,
                    )
                continue

            logger.info(
                "开始抓正文: book_id=%s title=%s pending=%s",
                book_id,
                book_row["title"],
                len(pending_rows),
            )

            for batch in chunked(list(pending_rows), batch_size):
                future_to_chapter = {
                    executor.submit(
                        client.fetch_chapter_content,
                        book_id,
                        int(row["chapter_id"]),
                    ): int(row["chapter_id"])
                    for row in batch
                }
                for future in concurrent.futures.as_completed(future_to_chapter):
                    chapter_id = future_to_chapter[future]
                    try:
                        chapter_record = future.result()
                        upsert_chapter_content(conn, chapter_record)
                        chapter_success_count += 1
                    except Exception as exc:
                        record_chapter_error(
                            conn,
                            book_id,
                            chapter_id,
                            str(exc),
                        )
                        chapter_failure_count += 1
                        logger.warning(
                            "正文抓取失败: book_id=%s chapter_id=%s error=%s",
                            book_id,
                            chapter_id,
                            exc,
                        )
                    finally:
                        chapter_processed_count += 1

                    if (
                        chapter_processed_count % chapter_progress_every == 0
                        or chapter_processed_count == pending_chapter_total
                    ):
                        logger.info(
                            (
                                "正文章节进度: %s/%s success=%s failure=%s "
                                "current_book=%s elapsed=%.2fs"
                            ),
                            chapter_processed_count,
                            pending_chapter_total,
                            chapter_success_count,
                            chapter_failure_count,
                            book_id,
                            time.time() - stage_start,
                        )

                refresh_book_content_progress(conn, book_id)

            completed_books += 1
            if book_index % progress_every == 0 or book_index == len(books):
                logger.info(
                    "正文书籍进度: %s/%s chapters=%s/%s elapsed=%.2fs",
                    completed_books,
                    len(books),
                    chapter_processed_count,
                    pending_chapter_total,
                    time.time() - stage_start,
                )


def print_stats(conn: sqlite3.Connection, db_path: str) -> None:
    book_row = conn.execute(
        """
        SELECT COUNT(*) AS total_books,
               COALESCE(
                   SUM(CASE WHEN chapter_count > 0 THEN 1 ELSE 0 END),
                   0
               ) AS catalog_ready_books,
               COALESCE(SUM(content_completed), 0) AS content_completed_books
        FROM books
        """
    ).fetchone()
    chapter_row = conn.execute(
        """
        SELECT COUNT(*) AS total_chapters,
               COALESCE(SUM(is_content_fetched), 0) AS fetched_chapters
        FROM chapters
        """
    ).fetchone()
    payload = {
        "db_path": os.path.abspath(db_path),
        "books": {
            "total": int(book_row["total_books"]),
            "catalog_ready": int(book_row["catalog_ready_books"]),
            "content_completed": int(book_row["content_completed_books"]),
        },
        "chapters": {
            "total": int(chapter_row["total_chapters"]),
            "fetched": int(chapter_row["fetched_chapters"]),
            "pending": int(chapter_row["total_chapters"])
            - int(chapter_row["fetched_chapters"]),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="起点搜书数据抓取：先抓目录入库，再抓章节正文入库。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_range_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--start", type=int, default=1)
        subparser.add_argument("--end", type=int, default=10000)
        subparser.add_argument("--db-path", default=DEFAULT_DB_PATH)
        subparser.add_argument("--timeout", type=int, default=20)
        subparser.add_argument("--retries", type=int, default=5)
        subparser.add_argument(
            "--sqlite-synchronous",
            default="FULL",
            choices=["FULL", "NORMAL"],
            help="SQLite 落盘强度，FULL 更抗断电，NORMAL 更快",
        )
        subparser.add_argument(
            "--min-request-interval",
            type=float,
            default=0.03,
            help="全局最小请求间隔秒数，避免瞬时突发",
        )
        subparser.add_argument(
            "--request-jitter",
            type=float,
            default=0.05,
            help="请求间隔随机抖动秒数，降低固定节奏特征",
        )
        subparser.add_argument(
            "--retry-backoff-base",
            type=float,
            default=1.5,
            help="重试退避基础秒数",
        )
        subparser.add_argument(
            "--retry-backoff-max",
            type=float,
            default=12.0,
            help="重试退避上限秒数",
        )
        subparser.add_argument("--progress-every", type=int, default=50)

    crawl_books_parser = subparsers.add_parser(
        "crawl-books", help="抓 1-N 书籍首页和目录到 SQLite"
    )
    add_common_range_args(crawl_books_parser)
    crawl_books_parser.add_argument("--concurrency", type=int, default=12)
    crawl_books_parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="已存在目录数据时仍强制刷新",
    )

    crawl_content_parser = subparsers.add_parser(
        "crawl-content", help="从 SQLite 中补全书籍章节正文"
    )
    add_common_range_args(crawl_content_parser)
    crawl_content_parser.add_argument("--concurrency", type=int, default=12)
    crawl_content_parser.add_argument(
        "--batch-size",
        type=int,
        default=120,
        help="每批提交的章节请求数",
    )
    crawl_content_parser.add_argument(
        "--chapter-progress-every",
        type=int,
        default=500,
        help="每处理多少章输出一次章节级进度",
    )
    crawl_content_parser.add_argument(
        "--max-pending-per-book",
        type=int,
        default=0,
        help="限制本次每本书最多抓多少个未完成章节，0 表示不限制",
    )

    sync_all_parser = subparsers.add_parser("sync-all", help="先抓目录再抓正文")
    add_common_range_args(sync_all_parser)
    sync_all_parser.add_argument("--concurrency", type=int, default=12)
    sync_all_parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="目录阶段强制刷新已有数据",
    )
    sync_all_parser.add_argument("--batch-size", type=int, default=120)
    sync_all_parser.add_argument(
        "--chapter-progress-every",
        type=int,
        default=500,
        help="每处理多少章输出一次章节级进度",
    )
    sync_all_parser.add_argument(
        "--max-pending-per-book",
        type=int,
        default=0,
        help="限制本次每本书最多抓多少个未完成章节，0 表示不限制",
    )

    stats_parser = subparsers.add_parser("stats", help="查看数据库抓取进度")
    stats_parser.add_argument("--db-path", default=DEFAULT_DB_PATH)

    return parser


def run_command(args: argparse.Namespace) -> None:
    command = args.command
    db_path = getattr(args, "db_path", DEFAULT_DB_PATH)
    synchronous_mode = getattr(args, "sqlite_synchronous", "FULL")
    conn = open_database(db_path, synchronous_mode=synchronous_mode)
    try:
        if command == "stats":
            print_stats(conn, db_path)
            return

        validate_range(args.start, args.end)
        client = BookApiClient(
            timeout=args.timeout,
            retries=args.retries,
            min_request_interval=args.min_request_interval,
            request_jitter=args.request_jitter,
            retry_backoff_base=args.retry_backoff_base,
            retry_backoff_max=args.retry_backoff_max,
        )

        if command == "crawl-books":
            crawl_books_stage(
                conn=conn,
                client=client,
                start=args.start,
                end=args.end,
                concurrency=args.concurrency,
                progress_every=args.progress_every,
                refresh_existing=args.refresh_existing,
            )
        elif command == "crawl-content":
            crawl_content_stage(
                conn=conn,
                client=client,
                start=args.start,
                end=args.end,
                concurrency=args.concurrency,
                progress_every=args.progress_every,
                batch_size=args.batch_size,
                chapter_progress_every=args.chapter_progress_every,
                max_pending_per_book=args.max_pending_per_book,
            )
        elif command == "sync-all":
            crawl_books_stage(
                conn=conn,
                client=client,
                start=args.start,
                end=args.end,
                concurrency=args.concurrency,
                progress_every=args.progress_every,
                refresh_existing=args.refresh_existing,
            )
            crawl_content_stage(
                conn=conn,
                client=client,
                start=args.start,
                end=args.end,
                concurrency=args.concurrency,
                progress_every=args.progress_every,
                batch_size=args.batch_size,
                chapter_progress_every=args.chapter_progress_every,
                max_pending_per_book=args.max_pending_per_book,
            )
        else:
            raise ValueError(f"不支持的命令: {command}")
    finally:
        conn.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_command(args)


if __name__ == "__main__":
    main()
