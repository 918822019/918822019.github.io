import re
import numpy as np
from typing import List, Tuple, Optional, Dict
from datasketch import MinHash, LeanMinHash


class TextDenoiserDeduplicator:
    """
    ============================================================================
    文本去噪去重处理类
    ============================================================================

    功能概述:
    --------
    1. 文本去噪（Denoising）:
       - 移除HTML/XML标签
       - 清理乱码和控制字符
       - 去除无意义的章节标题编号
       - 过滤作者求票、求收藏等营销话术
       - 移除纯数字行和过短的无意义行

    2. 段落级去重（Deduplication）:
       - 使用MinHash算法进行段落级相似度检测
       - 避免模型背诵重复的"名场面"
       - 保留首次出现的段落，移除后续高度相似内容

    设计理念:
    --------
    - 模块化设计：每个清理步骤独立，便于调试和扩展
    - 性能优化：预编译正则表达式，使用LeanMinHash节省内存
    - 可配置性：支持动态调整参数（阈值、n-gram大小等）
    - 详细统计：提供处理前后的对比数据和重复信息

    适用场景:
    --------
    - 网络小说数据清洗
    - 网页内容提取后处理
    - 大规模文本数据集预处理
    - 避免训练数据中的重复内容

    ============================================================================
    """

    def __init__(self,
                 minhash_threshold: float = 0.85,
                 num_perm: int = 128,
                 ngram_size: int = 3,
                 min_paragraph_length: int = 5):
        """
        ============================================================================
        初始化去噪去重器
        ============================================================================

        参数说明:
        --------
        minhash_threshold : float, 默认 0.85
            MinHash相似度阈值，范围[0, 1]
            - 值越大，去重越严格（相似度要求更高才会判定为重复）
            - 值越小，去重越宽松（较低相似度就会判定为重复）
            - 推荐范围：0.7 ~ 0.9
            - 示例：0.85表示相似度≥85%的段落会被判定为重复

        num_perm : int, 默认 128
            MinHash签名的排列数（哈希函数数量）
            - 影响MinHash的精度和内存占用
            - 值越大，精度越高，但内存和计算开销越大
            - 推荐范围：64 ~ 256
            - 128是精度和性能的良好平衡点

        ngram_size : int, 默认 3
            文本分词的n-gram大小
            - 对于中文文本，推荐使用2-3
            - 对于英文文本，推荐使用3-5
            - 值越大，对局部变化越敏感
            - 示例：ngram_size=3时，"你好世界" → ["你好世", "好世界"]

        min_paragraph_length : int, 默认 5
            最小段落长度阈值（字符数）
            - 长度小于此值的行会被过滤掉
            - 用于移除页码、章节号等无意义短行
            - 推荐范围：3 ~ 10

        ============================================================================

        初始化流程:
        -----------
        1. 保存配置参数
        2. 预编译所有正则表达式（提升性能）
        3. 准备去噪和去重所需的数据结构

        ============================================================================
        """
        # ========== 配置参数保存 ==========
        # MinHash相似度阈值，用于判定段落是否重复
        self.minhash_threshold = minhash_threshold

        # MinHash签名长度（哈希函数数量）
        self.num_perm = num_perm

        # n-gram大小，用于文本特征提取
        self.ngram_size = ngram_size

        # 最小段落长度阈值
        self.min_paragraph_length = min_paragraph_length

        # ========== 预编译正则表达式 ==========
        # 提升性能：避免每次处理时重复编译正则
        self._compile_patterns()

    # ============================================================================
    # 私有方法：正则表达式预编译
    # ============================================================================

    def _compile_patterns(self):
        """
        预编译所有正则表达式模式

        作用:
        ----
        1. 提升性能：避免在处理大量文本时重复编译正则
        2. 集中管理：所有正则模式在一个地方定义，便于维护
        3. 提高可读性：模式名称清晰表达其用途

        编译的模式包括:
        -------------
        1. HTML标签模式
        2. 章节标题模式（多种格式）
        3. 作者求票话术模式（多种表达）
        4. 纯数字行模式
        5. 段落分隔符模式

        注意:
        ----
        - 使用re.MULTILINE标志支持多行匹配
        - 使用re.IGNORECASE标志支持大小写不敏感匹配
        - 模式按处理顺序排列
        """
        # ========== 1. HTML标签清理模式 ==========
        # 匹配所有HTML标签：<tag>、</tag>、<tag attr="value">
        # 示例：<div class="content"> → 匹配整个标签
        self.html_tag_pattern = re.compile(r'<[^>]+>')

        # ========== 2. 控制字符清理模式 ==========
        # 匹配Unicode控制字符范围：
        # - \x00-\x1f: C0控制字符（ASCII控制字符）
        # - \x7f-\x9f: C1控制字符（扩展ASCII控制字符）
        # 这些字符通常为乱码或不可见字符
        self.control_char_pattern = re.compile(r'[\x00-\x1f\x7f-\x9f]')

        # ========== 3. 章节标题模式（多种格式） ==========
        # 支持多种章节标题格式：

        # 格式1: "第X章"、"第1章"、"第零章"等
        # - 支持中文数字：零一二三四五六七八九十百千
        # - 支持阿拉伯数字：0-9
        # - 后缀可以是：章、节、回、集、卷
        # - 可选的冒号分隔符：第1章： 或 第1章:
        self.chapter_patterns = [
            re.compile(r'^(第[零一二三四五六七八九十百千0-9]+[章节回集卷])\s*[:：]?\s*', re.MULTILINE),

            # 格式2: "Chapter X"、"CHAPTER 1"等（英文格式）
            # - 不区分大小写
            # - 支持空格分隔
            re.compile(r'^(chapter\s+\d+)\s*[:：]?\s*', re.MULTILINE | re.IGNORECASE),

            # 格式3: "第 1 章"（数字两侧有空格）
            re.compile(r'^(第\s+\d+\s+章)\s*[:：]?\s*', re.MULTILINE)
        ]

        # ========== 4. 作者求票话术模式（多种表达） ==========
        # 网络小说中常见的营销话术，用于请求读者支持
        # 模式库可扩展，覆盖主流小说平台的常见表达

        self.solicitation_patterns = [
            # 基础求票表达
            re.compile(r'求[收阅订阅][藏阅]|求推荐票|求月票|求打赏|求订阅|求支持|求收藏|新书求关注'),

            # 鼓励投票表达
            re.compile(r'觉得好看请投推荐票|喜欢本书请收藏|欢迎来到本书|欢迎追读|欢迎订阅'),

            # 本书求支持
            re.compile(r'本书求收藏|本书求推荐|本书求月票'),

            # 感谢支持
            re.compile(r'感谢支持|感谢订阅|感谢打赏|感谢投推荐票'),

            # 喜欢就支持
            re.compile(r'喜欢就收藏|喜欢请订阅|喜欢请投推荐票'),

            # 组合求票（常见于章节末尾）
            re.compile(r'求[收阅订阅][藏阅]求[推推][荐荐]票'),
            re.compile(r'求[收阅订阅][藏阅]，求[推推][荐荐]票'),

            # 点击操作
            re.compile(r'点击收藏|点击推荐|点击订阅'),

            # 收藏/推荐/订阅本书
            re.compile(r'收藏本书|推荐本书|订阅本书'),

            # 首订相关（针对连载小说）
            re.compile(r'求个首订|求首订支持|求保底月票')
        ]

        # ========== 5. 纯数字行模式 ==========
        # 匹配完全由数字组成的行
        # 用于过滤页码、行号等无意义内容
        # 示例：匹配 "123"、"001"，但不匹配 "第1章"
        self.pure_digit_pattern = re.compile(r'^\d+$')

        # ========== 6. 段落分隔符模式 ==========
        # 匹配两个或更多连续的换行符
        # 用于分割自然段落
        # 示例：\n\n 或 \n\n\n 都会被匹配
        self.paragraph_split_pattern = re.compile(r'\n{2,}')

    # ============================================================================
    # 公共方法：文本去噪（各步骤独立）
    # ============================================================================

    def clean_html_tags(self, text: str) -> str:
        """
        移除HTML/XML标签

        功能:
        ----
        删除文本中的所有HTML和XML标签，保留标签内的文本内容

        处理示例:
        --------
        输入: "<div><p>你好</p></div>"
        输出: "你好"

        输入: "<a href='link'>点击这里</a>"
        输出: "点击这里"

        参数:
        ----
        text : str
            原始文本，可能包含HTML标签

        返回:
        ----
        str
            移除所有HTML标签后的纯文本

        注意:
        ----
        - 此步骤应在其他清理步骤之前执行
        - 不会处理HTML实体（如 &nbsp;），如有需要可额外添加
        """
        return self.html_tag_pattern.sub('', text)

    def remove_control_chars(self, text: str) -> str:
        """
        移除控制字符和乱码

        功能:
        ----
        删除Unicode控制字符范围内的字符，这些通常是乱码或不可见字符

        清理范围:
        --------
        - C0控制字符：\x00-\x1f (ASCII控制字符)
        - C1控制字符：\x7f-\x9f (扩展ASCII控制字符)

        这些字符包括：
        - 空字符 (NULL)
        - 换行符、回车符（但保留\n用于段落分割）
        - 制表符（但保留\t，如有需要可额外处理）
        - 其他不可打印字符

        参数:
        ----
        text : str
            待处理的文本

        返回:
        ----
        str
            移除控制字符后的文本

        注意:
        ----
        - 此步骤会移除所有控制字符，包括一些可能有用的字符
        - 如需保留某些特殊字符，需修改正则表达式
        """
        return self.control_char_pattern.sub('', text)

    def remove_chapter_headers(self, text: str) -> str:
        """
        移除章节标题

        功能:
        ----
        删除文本开头的章节标题信息，如"第1章"、"Chapter 1"等

        支持的格式:
        ----------
        1. 中文格式：
           - "第1章"、"第零章"、"第100章"
           - "第1节"、"第1回"、"第1集"、"第1卷"
           - "第1章："、"第1章:"（带冒号）

        2. 英文格式：
           - "Chapter 1"、"CHAPTER 1"、"chapter 1"
           - "Chapter 1:"（带冒号）

        3. 带空格格式：
           - "第 1 章"

        处理示例:
        --------
        输入: "第1章：初入江湖"
        输出: "初入江湖"

        输入: "Chapter 1: The Beginning"
        输出: "The Beginning"

        参数:
        ----
        text : str
            待处理的文本

        返回:
        ----
        str
            移除章节标题后的文本

        注意:
        ----
        - 使用re.MULTILINE标志，支持多行文本中的章节标题
        - 只移除行首的章节标题（使用^锚点）
        - 保留章节标题后的实际内容
        """
        # 依次应用所有章节标题模式
        for pattern in self.chapter_patterns:
            text = pattern.sub('', text)
        return text

    def remove_solicitation_text(self, text: str) -> str:
        """
        移除作者求票话术

        功能:
        ----
        删除网络小说中常见的营销话术，如求收藏、求推荐票等

        清理的话术类型:
        --------------
        1. 直接求票：求收藏、求推荐票、求月票、求打赏
        2. 鼓励支持：觉得好看请投推荐票、喜欢本书请收藏
        3. 本书求支持：本书求收藏、本书求推荐
        4. 感谢表达：感谢支持、感谢订阅
        5. 组合求票：求收藏求推荐票
        6. 点击操作：点击收藏、点击推荐
        7. 首订相关：求个首订、求保底月票

        处理示例:
        --------
        输入: "求收藏！求推荐票！新书需要支持！"
        输出: "！新书需要支持！"

        输入: "觉得好看请投推荐票，感谢大家的支持！"
        输出: "，感谢大家的支持！"

        参数:
        ----
        text : str
            待处理的文本

        返回:
        ----
        str
            移除求票话术后的文本

        扩展建议:
        --------
        - 可根据具体平台特点添加更多话术模式
        - 可添加平台特定的营销用语
        - 可添加作者个人常用的求票表达
        """
        # 依次应用所有求票话术模式
        for pattern in self.solicitation_patterns:
            text = pattern.sub('', text)
        return text

    def filter_meaningless_lines(self, text: str) -> str:
        """
        过滤无意义行

        功能:
        ----
        移除文本中的无意义行，包括：
        1. 纯数字行（页码、行号等）
        2. 过短的行（可能为噪声）
        3. 空行（在后续步骤中处理）

        过滤规则:
        --------
        1. 跳过空行（保留用于段落分隔）
        2. 跳过完全由数字组成的行
        3. 跳过长度小于min_paragraph_length的行

        处理示例:
        --------
        输入:
        "这是有效内容

        123
        abc
        这也是有效内容"

        输出:
        "这是有效内容

        这也是有效内容"

        参数:
        ----
        text : str
            待处理的文本

        返回:
        ----
        str
            过滤无意义行后的文本

        注意:
        ----
        - 此步骤会改变行数，但保留段落结构
        - 空行会被保留（用于段落分隔）
        - 长度阈值可通过min_paragraph_length参数调整
        """
        lines = []
        # 逐行处理
        for line in text.split('\n'):
            line_stripped = line.strip()

            # 跳过空行（保留用于段落分隔）
            if not line_stripped:
                continue

            # 跳过纯数字行（页码、行号等）
            if self.pure_digit_pattern.fullmatch(line_stripped):
                continue

            # 跳过过短行（可能为噪声）
            if len(line_stripped) < self.min_paragraph_length:
                continue

            # 保留有效行
            lines.append(line_stripped)

        # 重新组合为文本
        return '\n'.join(lines)

    def merge_empty_lines(self, text: str) -> str:
        """
        合并连续空行

        功能:
        ----
        将多个连续的空行合并为单个空行，优化段落结构

        处理逻辑:
        --------
        1. 遍历所有行
        2. 遇到非空行：直接添加
        3. 遇到空行：只在前一行非空时添加（避免连续空行）
        4. 移除首尾的空行

        处理示例:
        --------
        输入:
        "第一段


        第二段"

        输出:
        "第一段

        第二段"

        参数:
        ----
        text : str
            待处理的文本

        返回:
        ----
        str
            合并连续空行后的文本

        优点:
        ----
        - 优化段落间距，避免过大的空白
        - 保持段落分隔的清晰性
        - 减少不必要的换行符
        """
        lines = text.split('\n')
        cleaned_lines = []
        prev_empty = False  # 标记前一行是否为空

        # 遍历所有行
        for line in lines:
            if line.strip():  # 非空行
                cleaned_lines.append(line)
                prev_empty = False
            elif not prev_empty:  # 第一个空行（前一行非空）
                cleaned_lines.append('')
                prev_empty = True

        # 移除首尾空行
        while cleaned_lines and not cleaned_lines[0].strip():
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()

        # 重新组合为文本
        return '\n'.join(cleaned_lines)

    # ============================================================================
    # 公共方法：完整去噪流程
    # ============================================================================

    def clean_text(self, text: str, verbose: bool = False) -> str:
        """
        ============================================================================
        完整的文本去噪处理
        ============================================================================

        功能概述:
        --------
        依次执行所有去噪步骤，返回清理后的文本

        处理流程:
        --------
        1. 移除HTML/XML标签
        2. 移除控制字符和乱码
        3. 移除章节标题
        4. 移除作者求票话术
        5. 过滤无意义行
        6. 合并连续空行

        参数:
        ----
        text : str
            原始文本，可能包含各种噪声

        verbose : bool, 默认 False
            是否打印详细的处理信息
            - True: 打印每一步的文本长度变化
            - False: 静默处理

        返回:
        ----
        str
            完全去噪后的文本

        处理示例:
        --------
        输入:
        "<div>
        <p>第1章：初入江湖</p>
        <p>青衫磊落险峰行，玉壁月华明。</p>
        <p>求收藏！求推荐票！</p>
        </div>"

        输出:
        "青衫磊落险峰行，玉壁月华明。"

        注意事项:
        --------
        1. 步骤顺序很重要，建议按当前顺序执行
        2. 每个步骤都是独立的，可根据需要跳过某些步骤
        3. 去噪后文本可能仍包含重复内容，需配合去重步骤使用

        ============================================================================
        """
        if verbose:
            # 打印原始文本信息
            original_length = len(text)
            print(f"【去噪开始】原始文本长度: {original_length} 字符")
            print(f"原始段落数: {len([p for p in text.split(chr(10) * 2) if p.strip()])}")
            print("-" * 60)

        # ========== 第1步：移除HTML/XML标签 ==========
        text = self.clean_html_tags(text)
        if verbose:
            print(f"✓ 步骤1 - 移除HTML标签后: {len(text)} 字符")

        # ========== 第2步：移除控制字符和乱码 ==========
        text = self.remove_control_chars(text)
        if verbose:
            print(f"✓ 步骤2 - 移除控制字符后: {len(text)} 字符")

        # ========== 第3步：移除章节标题 ==========
        text = self.remove_chapter_headers(text)
        if verbose:
            print(f"✓ 步骤3 - 移除章节标题后: {len(text)} 字符")

        # ========== 第4步：移除作者求票话术 ==========
        text = self.remove_solicitation_text(text)
        if verbose:
            print(f"✓ 步骤4 - 移除求票话术后: {len(text)} 字符")

        # ========== 第5步：过滤无意义行 ==========
        text = self.filter_meaningless_lines(text)
        if verbose:
            print(f"✓ 步骤5 - 过滤无意义行后: {len(text)} 字符")

        # ========== 第6步：合并连续空行 ==========
        text = self.merge_empty_lines(text)
        if verbose:
            print(f"✓ 步骤6 - 合并空行后: {len(text)} 字符")
            print("-" * 60)
            print(f"【去噪完成】最终文本长度: {len(text)} 字符")
            print(
                f"长度减少: {original_length - len(text)} 字符 ({(original_length - len(text)) / original_length * 100:.1f}%)")

        return text

    # ============================================================================
    # 私有方法：MinHash相关
    # ============================================================================

    def _text_to_ngrams(self, text: str) -> List[str]:
        """
        将文本转换为n-gram列表

        功能:
        ----
        将输入文本分割为连续的n-gram片段，用于MinHash特征提取

        n-gram原理:
        ---------
        n-gram是一种文本特征提取方法，将文本分割为连续的n个字符/词的组合

        示例（ngram_size=3）:
        -----------------
        输入: "你好世界"
        输出: ["你好世", "好世界"]

        输入: "abcdef"
        输出: ["abc", "bcd", "cde", "def"]

        参数:
        ----
        text : str
            待转换的文本

        返回:
        ----
        List[str]
            n-gram列表

        特殊处理:
        --------
        - 如果文本长度小于n-gram大小，返回包含整个文本的单元素列表
        - 如果文本为空，返回空列表

        注意:
        ----
        - 对于中文，推荐使用2-3
        - 对于英文，推荐使用3-5
        - n-gram大小可通过ngram_size参数调整
        """
        # 处理特殊情况
        if len(text) < self.ngram_size:
            return [text] if text else []

        # 生成n-gram列表
        ngrams = []
        for i in range(len(text) - self.ngram_size + 1):
            ngrams.append(text[i:i + self.ngram_size])

        return ngrams

    def _create_minhash(self, text: str) -> LeanMinHash:
        """
        为文本创建MinHash签名

        功能:
        ----
        使用MinHash算法为输入文本生成唯一的签名（哈希值），用于快速计算文本相似度

        MinHash原理:
        -----------
        MinHash是一种局部敏感哈希（LSH）算法，用于估计集合的Jaccard相似度

        工作流程:
        --------
        1. 将文本转换为n-gram集合
        2. 使用多个哈希函数对每个n-gram进行哈希
        3. 对每个哈希函数，保留最小的哈希值
        4. 所有最小哈希值组成MinHash签名

        优点:
        ----
        1. 快速：相比直接计算Jaccard相似度，速度快得多
        2. 内存友好：签名长度固定（num_perm），不随文本长度增长
        3. 准确：能够较好地估计文本相似度

        参数:
        ----
        text : str
            待生成签名的文本

        返回:
        ----
        LeanMinHash
            文本的MinHash签名（优化版，内存占用更小）

        技术细节:
        --------
        - 使用datasketch库的MinHash实现
        - 转换为LeanMinHash以减少内存占用
        - 每个n-gram被编码为UTF-8字节后进行哈希

        应用场景:
        --------
        - 段落级去重
        - 文本相似度检测
        - 重复内容识别
        """
        # 创建MinHash对象
        m = MinHash(num_perm=self.num_perm)

        # 将文本转换为n-gram
        ngrams = self._text_to_ngrams(text)

        # 为每个n-gram更新MinHash
        for ngram in ngrams:
            # 将n-gram编码为UTF-8字节，然后更新MinHash
            m.update(ngram.encode('utf-8'))

        # 转换为LeanMinHash以减少内存占用
        return LeanMinHash(m)

    # ============================================================================
    # 公共方法：段落级去重
    # ============================================================================

    def paragraph_level_dedup(self, text: str, verbose: bool = False) -> Tuple[str, List[Tuple[int, int, float]]]:
        """
        ============================================================================
        段落级MinHash去重
        ============================================================================

        功能概述:
        --------
        使用MinHash算法检测并移除文本中的重复段落，避免模型背诵重复的"名场面"

        核心算法:
        --------
        1. 使用MinHash为每个段落生成签名
        2. 计算段落之间的Jaccard相似度
        3. 相似度超过阈值的段落判定为重复
        4. 保留首次出现的段落，移除后续重复段落

        处理流程:
        --------
        1. 按段落分割文本（两个及以上换行符）
        2. 为每个段落创建MinHash签名
        3. 逐个比较段落相似度
        4. 保留唯一段落，记录重复信息
        5. 重新组合为去重后的文本

        参数:
        ----
        text : str
            已去噪的文本（建议先调用clean_text）

        verbose : bool, 默认 False
            是否打印详细的处理信息
            - True: 打印段落处理详情和重复信息
            - False: 静默处理

        返回:
        ----
        Tuple[str, List[Tuple[int, int, float]]]
        - 第一个元素：去重后的文本
        - 第二个元素：重复段落信息列表，每个元素为(原始索引, 保留索引, 相似度)

        返回值示例:
        ----------
        (
            "去重后的文本...",
            [(3, 0, 0.92), (5, 0, 0.88)]  # 段落3和5与段落0重复
        )

        去重策略:
        --------
        - 保留策略：保留首次出现的段落
        - 移除策略：移除后续所有相似度超过阈值的段落
        - 相似度计算：使用Jaccard相似度

        阈值说明:
        --------
        - 默认阈值：0.85（85%相似度）
        - 阈值越高，去重越严格（需要更高相似度才判定为重复）
        - 阈值越低，去重越宽松（较低相似度就会判定为重复）

        性能考虑:
        --------
        1. 使用LeanMinHash减少内存占用
        2. 时间复杂度：O(n²)，n为段落数
        3. 对于超长文本，建议分批处理

        注意事项:
        --------
        1. 建议先进行文本去噪，再进行去重
        2. 段落分割基于连续换行符，确保文本格式正确
        3. 对于极短段落，可能会影响去重效果

        ============================================================================
        """
        # ========== 第1步：分割段落 ==========
        # 使用预编译的正则表达式分割段落
        paragraphs = self.paragraph_split_pattern.split(text)

        # 清理段落：去除首尾空白，过滤空段落
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if verbose:
            print(f"\n【去重开始】段落总数: {len(paragraphs)}")

        # 处理边界情况
        if len(paragraphs) == 0:
            if verbose:
                print("【警告】文本中没有有效段落")
            return "", []

        # ========== 第2步：为每个段落创建MinHash签名 ==========
        if verbose:
            print("正在生成段落签名...")

        minhashes = []
        for para in paragraphs:
            minhashes.append(self._create_minhash(para))

        if verbose:
            print(f"✓ 完成 {len(minhashes)} 个段落的签名生成")

        # ========== 第3步：去重处理 ==========
        # 数据结构说明：
        # - unique_paragraphs: 保留的唯一段落列表
        # - unique_indices: 保留段落在原列表中的索引
        # - duplicates_info: 重复段落信息 [(原始索引, 保留索引, 相似度)]

        unique_paragraphs = []  # 存储唯一段落
        unique_indices = []  # 存储唯一段落的原始索引
        duplicates_info = []  # 存储重复信息

        if verbose:
            print("\n正在检测重复段落...")

        # 遍历所有段落
        for i, (para, mh) in enumerate(zip(paragraphs, minhashes)):
            is_duplicate = False  # 标记是否为重复段落
            duplicate_with = -1  # 记录与哪个段落重复
            similarity = 0.0  # 记录相似度

            # 与已保留的段落逐一比较
            for j, idx in enumerate(unique_indices):
                # 计算Jaccard相似度
                sim = mh.jaccard(minhashes[idx])

                # 如果相似度超过阈值，判定为重复
                if sim >= self.minhash_threshold:
                    is_duplicate = True
                    duplicate_with = idx
                    similarity = sim
                    break  # 找到重复即可退出

            # 处理重复段落
            if is_duplicate:
                duplicates_info.append((i, duplicate_with, similarity))
                if verbose:
                    print(f"  ⚠ 段落 {i} 与段落 {duplicate_with} 重复 (相似度: {similarity:.3f})")
            else:
                # 保留唯一段落
                unique_paragraphs.append(para)
                unique_indices.append(i)
                if verbose:
                    print(f"  ✓ 保留段落 {i}")

        # ========== 第4步：重新组合文本 ==========
        deduped_text = '\n\n'.join(unique_paragraphs)

        # 打印统计信息
        if verbose:
            print("\n" + "-" * 60)
            print(f"【去重完成】")
            print(f"  去重后段落数: {len(unique_paragraphs)}")
            print(f"  移除重复段落数: {len(duplicates_info)}")
            if duplicates_info:
                print(f"  重复率: {len(duplicates_info) / len(paragraphs) * 100:.1f}%")

        return deduped_text, duplicates_info

    # ============================================================================
    # 公共方法：完整处理流程
    # ============================================================================

    def process(self, text: str, verbose: bool = False) -> Dict:
        """
        ============================================================================
        完整处理流程：去噪 + 去重
        ============================================================================

        功能概述:
        --------
        一站式完成文本的去噪和去重处理，返回详细的处理结果和统计信息

        处理流程:
        --------
        1. 文本去噪（clean_text）
           - 移除HTML标签
           - 清理乱码
           - 移除章节标题
           - 过滤求票话术
           - 合并空行

        2. 段落级去重（paragraph_level_dedup）
           - 生成MinHash签名
           - 检测重复段落
           - 移除重复内容

        3. 统计分析
           - 长度变化
           - 段落数变化
           - 重复信息

        参数:
        ----
        text : str
            原始文本，可能包含各种噪声和重复内容

        verbose : bool, 默认 False
            是否打印详细的处理信息
            - True: 打印完整的处理日志和统计信息
            - False: 静默处理

        返回:
        ----
        Dict: 包含以下键值对

        {
            'cleaned_text': str,           # 去噪后的文本
            'deduped_text': str,           # 去重后的文本
            'duplicates_info': List,       # 重复段落信息 [(原索引, 保留索引, 相似度)]
            'stats': Dict                  # 统计信息
        }

        stats字典结构:
        -------------
        {
            'original_length': int,        # 原始文本长度（字符数）
            'cleaned_length': int,         # 去噪后长度
            'deduped_length': int,         # 去重后长度
            'original_paragraphs': int,    # 原始段落数
            'cleaned_paragraphs': int,     # 去噪后段落数
            'deduped_paragraphs': int,     # 去重后段落数
            'removed_duplicates': int      # 移除的重复段落数
        }

        使用示例:
        --------
        >>> processor = TextDenoiserDeduplicator()
        >>> result = processor.process(text, verbose=True)
        >>> cleaned = result['deduped_text']
        >>> stats = result['stats']
        >>> print(f"处理后长度: {stats['deduped_length']}")

        性能提示:
        --------
        1. 对于超长文本（>100MB），建议分批处理
        2. 可调整num_perm参数平衡精度和性能
        3. 可调整minhash_threshold控制去重严格程度

        错误处理:
        --------
        - 空文本：返回空结果，不抛出异常
        - 无有效段落：返回空文本，记录警告信息

        ============================================================================
        """
        # ========== 初始化统计信息 ==========
        stats = {
            'original_length': len(text),
            'cleaned_length': 0,
            'deduped_length': 0,
            'original_paragraphs': 0,
            'cleaned_paragraphs': 0,
            'deduped_paragraphs': 0,
            'removed_duplicates': 0
        }

        if verbose:
            print("=" * 70)
            print("开始文本去噪去重处理")
            print("=" * 70)

        # ========== 第1步：文本去噪 ==========
        if verbose:
            print("\n【阶段1：文本去噪】")
            print("-" * 70)

        cleaned_text = self.clean_text(text, verbose=verbose)
        stats['cleaned_length'] = len(cleaned_text)

        # 统计去噪后的段落数
        cleaned_paragraphs = self.paragraph_split_pattern.split(cleaned_text)
        cleaned_paragraphs = [p.strip() for p in cleaned_paragraphs if p.strip()]
        stats['cleaned_paragraphs'] = len(cleaned_paragraphs)

        # ========== 第2步：段落级去重 ==========
        if verbose:
            print("\n【阶段2：段落级去重】")
            print("-" * 70)

        deduped_text, duplicates_info = self.paragraph_level_dedup(cleaned_text, verbose=verbose)
        stats['deduped_length'] = len(deduped_text)

        # 统计去重后的段落数
        deduped_paragraphs = self.paragraph_split_pattern.split(deduped_text)
        deduped_paragraphs = [p.strip() for p in deduped_paragraphs if p.strip()]
        stats['deduped_paragraphs'] = len(deduped_paragraphs)
        stats['removed_duplicates'] = len(duplicates_info)

        # ========== 第3步：打印最终统计 ==========
        if verbose:
            print("\n" + "=" * 70)
            print("处理完成 - 统计摘要")
            print("=" * 70)
            print(f"{'指标':<20} {'原始':>12} {'去噪后':>12} {'去重后':>12}")
            print("-" * 70)
            print(f"{'文本长度(字符)':<20} {stats['original_length']:>12,} "
                  f"{stats['cleaned_length']:>12,} {stats['deduped_length']:>12,}")
            print(f"{'段落数':<20} {'-':>12} {stats['cleaned_paragraphs']:>12,} "
                  f"{stats['deduped_paragraphs']:>12,}")
            print("-" * 70)
            print(
                f"去噪减少: {(stats['original_length'] - stats['cleaned_length']) / stats['original_length'] * 100:.1f}%")
            print(
                f"去重减少: {(stats['cleaned_length'] - stats['deduped_length']) / stats['cleaned_length'] * 100:.1f}%")
            print(
                f"重复段落: {stats['removed_duplicates']} 个 ({stats['removed_duplicates'] / max(stats['cleaned_paragraphs'], 1) * 100:.1f}%)")
            print("=" * 70)

        # ========== 返回结果 ==========
        return {
            'cleaned_text': cleaned_text,
            'deduped_text': deduped_text,
            'duplicates_info': duplicates_info,
            'stats': stats
        }

    # ============================================================================
    # 公共方法：参数调整
    # ============================================================================

    def set_threshold(self, threshold: float):
        """
        动态设置MinHash相似度阈值

        功能:
        ----
        在不重新创建对象的情况下，动态调整去重的严格程度

        参数:
        ----
        threshold : float
            新的相似度阈值，范围[0, 1]
            - 0.0: 最宽松（任何相似都会判定为重复）
            - 1.0: 最严格（完全相同才判定为重复）
            - 推荐范围：0.7 ~ 0.9

        异常:
        ----
        ValueError
            如果阈值不在[0, 1]范围内

        使用示例:
        --------
        >>> processor = TextDenoiserDeduplicator()
        >>> processor.set_threshold(0.9)  # 更严格的去重
        >>> processor.set_threshold(0.7)  # 更宽松的去重

        应用场景:
        --------
        - 处理不同类型的文本（小说、新闻、技术文档）
        - 根据实际效果调整去重严格程度
        - 批量处理时动态调整参数
        """
        if 0 <= threshold <= 1:
            self.minhash_threshold = threshold
            if hasattr(self, '_verbose') and self._verbose:
                print(f"✓ MinHash阈值已更新为: {threshold}")
        else:
            raise ValueError("阈值必须在0到1之间")

    def set_ngram_size(self, size: int):
        """
        动态设置n-gram大小

        功能:
        ----
        在不重新创建对象的情况下，动态调整文本特征提取的粒度

        参数:
        ----
        size : int
            新的n-gram大小，必须大于0
            - 值越小：对局部变化越不敏感，可能漏检一些重复
            - 值越大：对局部变化越敏感，可能误判一些相似但不重复的内容
            - 中文推荐：2-3
            - 英文推荐：3-5

        异常:
        ----
        ValueError
            如果size小于等于0

        使用示例:
        --------
        >>> processor = TextDenoiserDeduplicator()
        >>> processor.set_ngram_size(2)  # 使用2-gram（适合短文本）
        >>> processor.set_ngram_size(4)  # 使用4-gram（更精细）

        影响:
        ----
        - 影响MinHash签名的生成
        - 影响相似度计算的精度
        - 需要重新处理已处理的文本才能生效
        """
        if size > 0:
            self.ngram_size = size
            if hasattr(self, '_verbose') and self._verbose:
                print(f"✓ n-gram大小已更新为: {size}")
        else:
            raise ValueError("n-gram大小必须大于0")

    # ============================================================================
    # 属性方法
    # ============================================================================

    @property
    def config(self) -> Dict:
        """
        获取当前配置信息

        返回:
        ----
        Dict: 包含所有配置参数的字典

        {
            'minhash_threshold': float,
            'num_perm': int,
            'ngram_size': int,
            'min_paragraph_length': int
        }

        使用示例:
        --------
        >>> processor = TextDenoiserDeduplicator()
        >>> print(processor.config)
        {
            'minhash_threshold': 0.85,
            'num_perm': 128,
            'ngram_size': 3,
            'min_paragraph_length': 5
        }
        """
        return {
            'minhash_threshold': self.minhash_threshold,
            'num_perm': self.num_perm,
            'ngram_size': self.ngram_size,
            'min_paragraph_length': self.min_paragraph_length
        }


# ============================================================================
# 使用示例和测试代码
# ============================================================================

if __name__ == "__main__":
    # ========== 示例文本 ==========
    sample_text = """
    <div class="novel-content">
    <h1>第1章：初入江湖</h1>
    <p>青衫磊落险峰行，玉壁月华明。</p>
    <p>这是一个经典的武侠场景描写，青衫客独行险峰，月光如水洒在玉壁之上。</p>

    <p>求收藏！求推荐票！新书需要大家支持！</p>

    <p>青衫磊落险峰行，玉壁月华明。</p>
    <p>这是一个经典的武侠场景描写，青衫客独行险峰，月光如水洒在玉壁之上。</p>

    <p>第2章：江湖险恶</p>
    <p>江湖路远，人心难测。行走在江湖中，需要时刻保持警惕。</p>

    <p>123
    456</p>

    <p>青衫磊落险峰行，玉壁月华明。</p>
    <p>这是一个经典的武侠场景描写，青衫客独行险峰，月光如水洒在玉壁之上。</p>

    <p>喜欢本书请投推荐票，感谢支持！点击收藏不迷路！</p>

    <p>第3章：侠之大者</p>
    <p>侠之大者，为国为民。真正的侠客，心怀天下。</p>
    </div>
    """

    print("=" * 80)
    print("TextDenoiserDeduplicator 类使用示例")
    print("=" * 80)

    # ========== 创建处理器实例 ==========
    print("\n【步骤1】创建处理器实例")
    print("-" * 80)
    processor = TextDenoiserDeduplicator(
        minhash_threshold=0.85,  # 相似度阈值85%
        num_perm=128,  # MinHash签名长度128
        ngram_size=3,  # 使用3-gram
        min_paragraph_length=5  # 最小段落长度5字符
    )
    print(f"✓ 处理器创建成功")
    print(f"  配置信息: {processor.config}")

    # ========== 执行完整处理 ==========
    print("\n【步骤2】执行完整处理（去噪 + 去重）")
    print("-" * 80)
    result = processor.process(sample_text, verbose=True)

    # ========== 查看处理结果 ==========
    print("\n【步骤3】查看处理结果")
    print("=" * 80)
    print("\n【去重后的最终文本】")
    print("-" * 80)
    print(result['deduped_text'])

    print("\n" + "=" * 80)
    print("【重复段落详细信息】")
    print("=" * 80)
    if result['duplicates_info']:
        print(f"共发现 {len(result['duplicates_info'])} 处重复:")
        for i, (orig_idx, keep_idx, sim) in enumerate(result['duplicates_info'], 1):
            print(f"  {i}. 段落 {orig_idx} 与 段落 {keep_idx} 重复 (相似度: {sim:.3f})")
    else:
        print("未发现重复段落")

    print("\n" + "=" * 80)
    print("【统计信息】")
    print("=" * 80)
    stats = result['stats']
    print(f"原始文本长度: {stats['original_length']:,} 字符")
    print(f"去噪后长度:   {stats['cleaned_length']:,} 字符 "
          f"(减少 {(stats['original_length'] - stats['cleaned_length']) / stats['original_length'] * 100:.1f}%)")
    print(f"去重后长度:   {stats['deduped_length']:,} 字符 "
          f"(减少 {(stats['cleaned_length'] - stats['deduped_length']) / stats['cleaned_length'] * 100:.1f}%)")
    print(f"段落数变化:   {stats['cleaned_paragraphs']} → {stats['deduped_paragraphs']} "
          f"(移除 {stats['removed_duplicates']} 个重复段落)")

    print("\n" + "=" * 80)
    print("示例结束")
    print("=" * 80)