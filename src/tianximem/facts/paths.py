"""Compile supported literal question forms into ordered evidence dependencies.

Entity names remain verbatim; uncertain or unparsed forms do not short-circuit
Search. This module does not extract source facts or call a model.
"""

import re

from tianximem.facts.evidence import PROFILE_FIELDS, relation_key

#: 档案字段白名单——**唯一一份**在 [`evidence.PROFILE_FIELDS`](./evidence.py)。
#: ⚠ 抽取侧与查询侧都从那里取：两边对不上时，题会**静默匹配不到**。
_PROFILE_FIELD_NAMES = frozenset(relation_key(name) for name in PROFILE_FIELDS)

RELATION_NAMES = {
    "author": ("author", "writer"),
    "capital": ("capital", "capital city"),
    "country of citizenship": ("country of citizenship", "nationality", "citizenship"),
    "country of origin": ("country of origin",),
    "creator": ("creator",),
    "spouse": ("spouse", "husband", "wife", "partner"),
    "founded by": ("founder", "founded by"),
    "educated at": ("educated at", "alma mater"),
    "employee": ("employer", "workplace"),
    "head of state": ("head of state", "head of the state"),
    "head of government": ("head of government", "head of the government"),
    "chief executive officer": ("chief executive officer", "CEO"),
    "headquarters location": ("headquarters location", "headquarters", "main office"),
    "original broadcaster": ("original broadcaster",),
    "developer": ("developer",),
    "manufacturer": ("manufacturer", "producer"),
    "director / manager": ("director", "manager", "director / manager"),
    "chairperson": ("chairperson",),
    "place of birth": ("place of birth", "birthplace", "city of birth"),
    "date of birth": ("date of birth", "birth date", "birthday"),
    "place of death": ("place of death", "location of death"),
    "location of formation": ("location of formation",),
    "languages spoken, written or signed": ("language spoken", "native language", "native tongue"),
    "official language": ("official language",),
    "notable work": (
        "notable work",
        "notable achievement",
        "claim to fame",
        "notable work or achievement",
    ),
    "religion or worldview": ("religion", "faith", "religious affiliation", "worldview"),
    "continent": ("continent",),
    "sport": ("sport",),
    "performer": ("performer",),
    "head coach": ("head coach",),
    "position played on team / speciality": ("position played on team / speciality",),
    "language of work or name": ("language of work or name",),
    # ⚠ **不要单独给 `field of work` 建关系名**：官方语料把它存成 `occupation`
    # （MQuAKE 的 P106），题面说 "field of work"——两者必须是**同一个**规范关系，
    # 否则计划走 `field of work`、事实在 `occupation`，永远 `unsupported_walk`
    # （2026-10-19 实测）。别名由下面 `occupation` 那一行统一收。
    "work location": ("work location", "workplace", "work place"),
    "genre": ("genre", "type of music"),
    "occupation": ("occupation", "field of work", "profession"),
    "child": ("child",),
    "officeholder": ("officeholder",),
}


_ALIASES = sorted(
    ((a, rel) for rel, names in RELATION_NAMES.items() for a in names), key=lambda p: -len(p[0])
)


def noun_path(text: str, depth: int = 0) -> tuple[str, tuple[str, ...]] | None:
    """A nested noun phrase is a literal root followed by ordered dependencies."""
    # Parsing recognizes longer paths so the executor can reject them as a
    # whole at its configured hop limit. A parser miss must not degrade into
    # legacy relation-keyword retrieval of only the first hop.
    if depth > 32:
        return None
    literal = text.strip()
    value = re.sub(r"^(?:(?:the|a|an|current)\s+)+", "", literal, flags=re.I)
    for alias, rel in _ALIASES:
        m = re.fullmatch(re.escape(alias) + r" (?:of|in) (.+)", value, re.I)
        if m:
            nested = noun_path(m[1], depth + 1)
            return (nested[0], (*nested[1], rel)) if nested else None
    citizen = re.fullmatch(
        r"(?:the )?(?:country|nation)(?: located(?: in)?)?,? (?:where|that) (.+?) "
        r"(?:is a citizen(?: of)?|holds citizenship|is a national of)",
        value,
        re.I,
    )
    if citizen:
        nested = noun_path(citizen[1], depth + 1)
        return (nested[0], (*nested[1], "country of citizenship")) if nested else None
    # 反向所有格："the country located, whose citizen is X"。
    citizen_reversed = re.fullmatch(
        r"(?:the )?(?:country|nation)(?: located)?,? whose citizen(?:ry)? is (.+)", value, re.I
    )
    if citizen_reversed:
        nested = noun_path(citizen_reversed[1], depth + 1)
        return (nested[0], (*nested[1], "country of citizenship")) if nested else None
    relative_frames = (
        ("author", r"(?:person|individual) who (?:wrote|authored) (.+)"),
        ("creator", r"(?:person|individual) who created (.+)"),
        ("developer", r"(?:company|organization) that developed (.+)"),
        ("employee", r"(?:company|organization) that employs (.+)"),
        ("educated at", r"(?:university|college|school) that educated (.+)"),
        ("educated at", r"(?:university|college|school) (?:located,? )?where (.+?) (?:was educated|received their education|studied|attended)"),
        ("educated at", r"(?:university|college|school) that (.+?) attended"),
        ("sport", r"sport associated with (.+)"),
        ("spouse", r"person (?:who is )?married to (.+)"),
        # **"the <关系> of X" 的从句式**（官方 MQuAKE，2026-10-19）：
        # "the country whose citizen performed X" / "the city where the CEO of
        # the developer of X was born"。都是「外层实体 + 内层从句」，通用嵌套。
        # `whose citizen performed X`：先沿该动词的关系走到 X，再到公民国。
        (("performer", "country of citizenship"), r"(?:country|nation) whose citizen performed (.+)"),
        (("creator", "country of citizenship"), r"(?:country|nation) whose citizen created (.+)"),
        (("founded by", "country of citizenship"), r"(?:country|nation) whose citizen founded (.+)"),
        (("author", "country of citizenship"), r"(?:country|nation) whose citizen wrote (.+)"),
        (("director / manager", "country of citizenship"), r"(?:country|nation) whose citizen directed (.+)"),
        (("performer", "country of citizenship"), r"(?:country|nation) whose citizen played (.+)"),
        ("country of citizenship", r"(?:country|nation) whose citizen is (.+)"),
        ("place of birth", r"(?:city|town) (?:where|in which) the CEO of the developer of (.+?) was born"),
        # **谓语后置的裸从句**（疑问壳已经剥掉了外层名词，官方 MQuAKE）：
        # "… the sport developed, that is associated with X" 的内层是
        # "is associated with the position played by X"——方向是「内层实体 → 外层属性」。
        # ⚠ 只收**带谓语标记**的形态；`(.+?) is renowned for` 这种会把普通名词短语
        # 也吞掉（实测 MQuAKE 815 → 782），不放进来。
        ("sport", r"(?:is |was )?(?:associated with|linked to|related to) (.+)"),
        ("position played on team / speciality", r"(?:is |was )?(?:played by|held by) (.+)"),
        ("country of origin", r"(?:created|developed|originated|produced) the sport (?:that is |which is )?(?:associated with|linked to|played by) (.+)"),
        ("country of origin", r"gave birth to the sport (?:that is |which is )?(?:associated with|linked to|played by) (.+)"),
        ("country of origin", r"(?:created|developed|originated|produced) the (?:music genre|genre) (?:played by|associated with) (.+)"),
        ("place of birth", r"(?:city|town) (?:where|in which) (.+?) was born"),
        ("place of birth", r"(?:city|town) (?:located|situated) (?:where|in which|in,? where) (.+?) was born"),
        (
            ("genre", "country of origin"),
            r"(?:country|nation) (?:located|situated)?,? (?:that|which|in,? where|in that|where) "
            r"(?:originated|created|developed) the (?:music genre|genre of music|type of music|music style|style of music) "
            r"(?:played by|associated with) (.+)",
        ),
        (
            ("sport", "country of origin"),
            r"(?:country|nation) (?:located|situated)?,? (?:that|which|in,? where|in that|where) "
            r"(?:originated|created|developed) the sport "
            r"(?:played by|associated with) (.+)",
        ),
        # "X is renowned/famous/celebrated for" —— 内层是「X 的 notable work」。
        # ⚠ 只收**完整的 `X is … for` 短语**（全串匹配），不做子串匹配。
        ("notable work", r"(.+?) (?:is|was) (?:renowned|famous|celebrated|known) for"),
        (
            ("sport", "country of origin"),
            r"(?:country|nation) (?:located|situated) (?:that|which|in,? where|in that|where) "
            r"the sport (?:that is |which is )?(?:associated with|linked to) (.+?) was created",
        ),
        # "是哪个大陆，位于'其运动与 X 相关'的国家"——`country of origin` 之外还要
        # **先沿 `sport` 走一步**（否则根会停在 "… was created" 这样的从句上）。
        (
            ("sport", "country of origin"),
            r"(?:country|nation) (?:located|situated) (?:in,? where|in that|where) "
            r"(?:the sport )?(?:that is |which is )?(?:associated with|linked to|played by) (.+?) "
            r"(?:was created|was developed|originated)",
        ),
    )
    for relative_relation, pattern in relative_frames:
        relative = re.fullmatch(pattern, value, re.I)
        if relative:
            nested = noun_path(relative[1], depth + 1)
            if nested:
                # rel 可以是**有序的几步**（`the country that originated the music genre
                # played by X` ⇒ genre → country of origin），不是业务分支。
                return (
                    nested[0],
                    (*nested[1], *(relative_relation if isinstance(relative_relation, tuple) else (relative_relation,))),
                )
    producer = re.fullmatch(r"company that (?:produced|manufactured) (.+)", value, re.I)
    if producer:
        nested = noun_path(producer[1], depth + 1)
        return (nested[0], (*nested[1], "manufacturer")) if nested else None
    origin = re.fullmatch(
        r"country where (.+?) (?:originated|was created|was produced)", value, re.I
    )
    if origin:
        nested = noun_path(origin[1], depth + 1)
        return (nested[0], (*nested[1], "country of origin")) if nested else None
    # "the country that created X" / "the country where X was developed"
    # （MQuAKE 的原创国问法）：原文事实字段是 `country of origin`。
    created = re.fullmatch(
        r"(?:the )?(?:country|nation) (?:that|which) (?:created|developed|produced) (.+)",
        value,
        re.I,
    )
    if created:
        nested = noun_path(created[1], depth + 1)
        return (nested[0], (*nested[1], "country of origin")) if nested else None
    developed = re.fullmatch(r"(?:the )?(?:country|nation) where (.+?) was developed", value, re.I)
    if developed:
        nested = noun_path(developed[1], depth + 1)
        return (nested[0], (*nested[1], "country of origin")) if nested else None
    # "the city where the headquarters of X is located" / "the city that serves as
    # the headquarters of X"——同样是通用的 `headquarters location` 嵌套（MQuAKE）。
    city_hq = re.fullmatch(
        r"(?:the )?(?:city|town) (?:where|that) (?:the )?headquarters of (.+?) "
        r"(?:is|are) located",
        value,
        re.I,
    )
    if city_hq is None:
        city_hq = re.fullmatch(
            r"(?:the )?(?:city|town) that serves as the headquarters of (.+)", value, re.I
        )
    if city_hq:
        nested = noun_path(city_hq[1], depth + 1)
        return (nested[0], (*nested[1], "headquarters location")) if nested else None
    # **描述性名词短语**（官方 MQuAKE 的主力问法，2026-10-18）：根不是字面名字，而是
    # "the company responsible for developing X" / "the music style played by X" /
    # "the country whose citizen performed X" 这类**原文里就有的关系短语**。
    # 与上面那些框架同一口径：都是「外层属性 + 内层实体」的嵌套，不是业务分支。
    descriptive = (
        (
            ("developer",),
            r"(?:the )?(?:company|organization|business) (?:responsible for|that) "
            r"(?:developing|developed|produces|produced) (.+)",
        ),
        # **"created" 对齐 `creator`，不是 `developer`**（2026-10-19）：MQuAKE 的
        # "the company that created Portal" 在语料里是 `creator` 边；映射成 developer
        # 会让整条 walk 走不通（实测 21 题）。
        (
            ("creator",),
            r"(?:the )?(?:company|organization|business) (?:that )?created (.+)",
        ),
        (
            ("developer",),
            r"(?:the )?(?:company|organization) that (?:is )?responsible for (.+)",
        ),
        (("performer",), r"(?:the )?(?:music style|genre|song|track) (?:played|performed) by (.+)"),
        (("performer",), r"(?:the )?(?:person|artist) who (?:played|performed) (.+)"),
        (
            ("country of citizenship",),
            r"(?:the )?(?:country|nation) whose citizen (?:is |was )?(.+)",
        ),
        (("country of citizenship",), r"(?:the )?citizen of (?:the )?(?:country|nation) (.+)"),
        (("creator",), r"(?:the )?(?:person|artist|author) (?:who |that )?created (.+)"),
        (("author",), r"(?:the )?(?:person|author) (?:who |that )?wrote (.+)"),
        (
            ("head of state",),
            r"(?:the )?(?:current )?head of (?:the )?(?:state) of "
            r"(?:the )?(.+?) (?:government|state)",
        ),
        # "the current head of the Zimbabwe government"（国家在中间）——**注意是
        # `head of government`**：MQuAKE 的 `head of the X government` 问法里 gold
        # 是政府首脑（P6），不是国家元首（P35）。实测 "What is the profession of the
        # current head of the Zimbabwe government?" 的语料边是 `head of government`
        # （2026-10-19）。
        (
            ("head of government",),
            r"(?:the )?(?:current )?head of (?:the )?(?:government) of (?:the )?(.+)",
        ),
        (
            ("head of government",),
            r"(?:the )?(?:current )?head of the (.+?) (?:government)",
        ),
        (
            ("head of state",),
            r"(?:the )?(?:current )?head of (?:the )?(?:state) of (?:the )?(.+)",
        ),
        (
            ("head of state",),
            r"(?:the )?(?:current )?head of the (.+?) (?:state)",
        ),
    )
    relations: tuple[str, ...]
    for relations, pattern in descriptive:
        match = re.fullmatch(pattern, value, re.I)
        if match:
            nested = noun_path(match[1], depth + 1)
            if nested:
                return (nested[0], (*nested[1], *relations))
    # "the capital/continent/language … of the country that/where X is a citizen"
    # （MQuAKE 那批两跳问法；都是**通用**的「属性 of 国家」嵌套，不是业务专用分支）。
    country_attr = re.fullmatch(
        r"(?:the\s+)?(capital|continent|official language|head of state|head of government|"
        r"leader(?: of the (?:state|government))?)"
        r" (?:city |language )?(?:of|in) (?:the )?(?:country|nation) "
        r"(?:that|where) (.+?) (?:is a citizen(?: of)?|holds citizenship|is a national of)",
        value,
        re.I,
    )
    if country_attr:
        # "current leader" 只是同一字段的时间限定（`_current` 负责取舍），映射到原文里
        # 同义的状态字段，不另造字段。⚠ MQuAKE 的 `leader of the country` 问法里
        # **21/22 的 gold 是 head of government**（2026-10-17 实测），所以映射到
        # `head of government`；只有 `leader of the state` 才映射到 `head of state`。
        relation = (
            "head of government"
            if country_attr[1].lower() in {"leader", "leader of the government"}
            else "head of state"
            if country_attr[1].lower() == "leader of the state"
            else country_attr[1]
        )
        nested = noun_path(country_attr[2], depth + 1)
        return (nested[0], (*nested[1], "country of citizenship", relation)) if nested else None
    named_office = re.fullmatch(r"entity known as (.+)", value, re.I)
    if named_office and re.search(
        r"\b(?:secretary[- ]general|president|governor|chairperson|CEO|chief .+ officer)$",
        named_office[1],
        re.I,
    ):
        return named_office[1], ("officeholder",)
    # **谓语后置的相对从句**（官方 MQuAKE 的另一半问法，2026-10-19）：
    # "the sport that is associated with the position played by X" /
    # "the sport linked to the position played by X" / "the work that Harper Lee
    # is renowned for" / "the city located where X died"。都是**通用嵌套**：
    # 外层属性 + 内层实体，不新增业务分支。
    # ⚠ 只收"关系短语 + 内层名词短语"，**不收**任何疑问词——那是 `question_path` 的活。
    postposed = (
        # the sport (that is) (associated with|linked to|connected to|involving|involves|related to) X
        (
            ("sport",),
            r"(?:the )?sport (?:that is |which is )?(?:associated with|linked to|connected to|"
            r"involves?|involving|related to|that involves?) (.+)",
        ),
        # the position (that) X plays / the position played by X
        (
            ("position played on team / speciality",),
            r"(?:the )?position (?:that |which )?(?:is )?(?:played by|played for|held by|"
            r"of|speciality of) (.+)",
        ),
        (
            ("position played on team / speciality",),
            r"(?:the )?position (?:that |which )?(.+?) plays",
        ),
        # the sport (that) X plays / the sport X is associated with
        (("sport",), r"(?:the )?sport (?:that |which )?(.+?) (?:plays|is associated with)"),
        # the work (that) X is renowned/famous/celebrated for
        (
            ("notable work",),
            r"(?:the )?(?:work|book|novel|song|film|movie) (?:that |which )?(.+?) "
            r"(?:is renowned for|is famous for|is celebrated for|is known for|wrote|authored|"
            r"created|composed|directed)",
        ),
        # the work written that X is renowned for（谓语在前、从句在后）
        (
            ("notable work",),
            r"(?:the )?(?:work|book|novel|song|film|movie) (?:\w+ )?(?:that|which) (.+?) "
            r"(?:is renowned for|is famous for|is celebrated for|is known for)",
        ),
        # the city where X died / the city located where X died / the place X died
        (
            ("place of death",),
            r"(?:the )?(?:city|town|place|location) (?:located |situated )?(?:where|in which) "
            r"(.+?) (?:died|passed away|met their demise)",
        ),
        # the city where X was born
        (
            ("place of birth",),
            r"(?:the )?(?:city|town|place|location) (?:located |situated )?(?:where|in which) "
            r"(.+?) was born",
        ),
        # the country that/which is home to X
        (("country of citizenship",), r"(?:the )?(?:country|nation) (?:that|which) is home to (.+)"),
        # the country that/which gave birth to X
        (
            ("country of origin",),
            r"(?:the )?(?:country|nation) (?:that|which) gave birth to (.+)",
        ),
        # the entity/group that performed X
        (
            ("performer",),
            r"(?:the )?(?:entity|group|band|artist|person|organization) (?:that|which) "
            r"(?:performed|plays|sang|recorded) (.+)",
        ),
        # the religion associated with X / the religion tied to X
        (
            ("religion or worldview",),
            r"(?:the )?(?:religion|faith) (?:that is |which is )?(?:associated with|affiliated with|"
            r"connected to|tied to|linked to) (.+)",
        ),
        # the workplace of X / the city where X operated
        (("work location",), r"(?:the )?workplace of (.+)"),
        (("work location",), r"(?:the )?city where (.+?) (?:operated|worked|lived)"),
        # the country that X is from / the country X is a national of
        (("country of citizenship",), r"(?:the )?(?:country|nation) (?:that |which )?(.+?) is from"),
        (
            ("country of citizenship",),
            r"(?:the )?(?:country|nation) (?:that |which )?(.+?) is a national of",
        ),
        (
            ("country of citizenship",),
            r"(?:the )?(?:country|nation) of which (.+?) is a national",
        ),
        # "the country of the creator of X" / "the country of X"
        (
            ("country of citizenship",),
            r"(?:the )?(?:country|nation) of the (creator|author|founder|director|performer) "
            r"of (.+)",
        ),
        (
            ("country of origin",),
            r"(?:the )?(?:country|nation) (?:that|which) gave birth to the sport "
            r"(?:that is |which is )?(?:associated with|linked to) (.+)",
        ),
        (
            ("country of origin",),
            r"(?:the )?(?:country|nation) (?:that|which) (?:originated|created|developed) "
            r"the (?:music genre|genre|type of music) (?:played by|associated with) (.+)",
        ),
        (
            ("country of origin",),
            r"(?:the )?(?:country|nation) (?:that|which) gave birth to the (?:music genre|genre) "
            r"(?:played by|associated with) (.+)",
        ),
        (
            ("country of citizenship",),
            r"(?:the )?(?:country|nation) (?:that |which )?(.+?) holds? (?:a )?citizenship",
        ),
        # the field (that) X works in / is involved in
        (("occupation",), r"(?:the )?field (?:that |which )?(.+?) (?:works? in|is involved in)"),
        (("occupation",), r"(?:the )?field in which (.+?) (?:works?|is involved)"),
        # the citizen who created X —— 国家嵌套里的一层（`country that is home to the citizen who …`）
        (
            ("creator",),
            r"(?:the )?citizen (?:who|that) (?:created|founded|wrote|authored) (.+)",
        ),
        # the university's headquarters …, which is the alma mater of X
        (
            ("educated at", "headquarters location"),
            r"(?:the )?university(?:'s)? headquarters (?:located,? )?(?:which|that) "
            r"(?:is )?the alma mater of (.+)",
        ),
        (
            ("educated at", "headquarters location"),
            r"(?:the )?headquarters of (?:the )?university,? (?:which|that) (?:is )?"
            r"the alma mater of (.+)",
        ),
        # "the country that developed the sport associated with X"
        (
            ("sport", "country of origin"),
            r"(?:the )?(?:country|nation) (?:that|which) "
            r"(?:developed|created|originated|produced) the sport (?:that is |which is )?"
            r"(?:associated with|linked to|related to|played by) (.+)",
        ),
        # the type of music played by X / the music genre played by X
        (
            ("genre",),
            r"(?:the )?(?:type of music|music genre|genre of music|music style|style of music) "
            r"(?:that is |which is )?(?:played by|associated with|performed by) (.+)",
        ),
        # the field of work of X / the field X works in
        (("occupation",), r"(?:the )?field of work of (.+)"),
        (("occupation",), r"(?:the )?field (?:that |which )?(.+?) works? in"),
        # the language of the work that X ...
        (
            ("language of work or name",),
            r"(?:the )?(?:language|original language) (?:of|used in|written in) "
            r"(?:the )?(?:work|book|novel|song|film) (?:that |which )?(.+)",
        ),
    )
    # **主语后置、谓语夹在中间的从句**（官方 MQuAKE，2026-10-19）：
    # "the founder of the religion tied to X" 的 founder 在内、religion 在中间。
    # 先把 "the <属性> of the <东西> <从句> X" 拆成「内层实体 X → 外层属性」两步。
    inner_outer = (
        (
            ("religion or worldview", "founded by"),
            r"(?:the )?founder of the religion (?:that is |which is )?"
            r"(?:associated with|tied to|connected to|affiliated with|linked to) (.+)",
        ),
        (
            ("religion or worldview", "founded by"),
            r"(?:the )?founder of (?:the )?(?:religion|faith) (.+)",
        ),
        (
            ("educated at", "headquarters location"),
            r"(?:the )?headquarters of the university (?:that |which )?(?:is )?"
            r"(?:located at|of) (.+)",
        ),
        (
            ("country of citizenship", "capital"),
            r"(?:the )?capital of the country (?:that |which )?(?:is )?(?:home to|hosts) (.+)",
        ),
        (
            ("country of citizenship", "continent"),
            r"(?:the )?continent of the country (?:that |which )?(?:is )?(?:home to|hosts) (.+)",
        ),
        (
            ("country of origin", "continent"),
            r"(?:the )?continent of origin of the (?:sport|work|music) "
            r"(?:that is |which is )?(?:associated with|played by|linked to) (.+)",
        ),
    )
    for relations, pattern in inner_outer:
        match = re.fullmatch(pattern, value, re.I)
        if match:
            if len(match.groups()) > 1:
                # 形如 `the country of the <关系> of X`：组 1 是内层关系词、组 2 是实体。
                nested = noun_path(match[2], depth + 1)
                if nested:
                    return (
                        nested[0],
                        (*nested[1], relation_key(match[1]), *relations),
                    )
                continue
            nested = noun_path(match[1], depth + 1)
            if nested:
                return (nested[0], (*nested[1], *relations))
    for relations, pattern in postposed:
        match = re.fullmatch(pattern, value, re.I)
        if match:
            if len(match.groups()) > 1:
                # 两组：`the country of the <关系> of X`（组 1 关系词、组 2 实体）。
                nested = noun_path(match[2], depth + 1)
                if nested:
                    return (nested[0], (*nested[1], relation_key(match[1]), *relations))
                continue
            nested = noun_path(match[1], depth + 1)
            if nested:
                return (nested[0], (*nested[1], *relations))
    # "the country where the sport associated with X was created/developed/originated" /
    # "the country that created the sport related to the field X works in"
    sport_related = re.fullmatch(
        r"(?:the )?(?:country|nation) (?:that|which) (?:created|developed|produced|originated) "
        r"the sport (?:that is |which is )?(?:associated with|linked to|connected to|related to|"
        r"played by|involves?) (.+)",
        value,
        re.I,
    )
    if sport_related:
        nested = noun_path(sport_related[1], depth + 1)
        return (nested[0], (*nested[1], "sport", "country of origin")) if nested else None
    # "the country whose citizen created the leader of X" —— 国家 → 公民 → 关系
    citizen_created = re.fullmatch(
        r"(?:the )?(?:country|nation) whose citizen (?:created|founded|wrote|authored) (.+)",
        value,
        re.I,
    )
    if citizen_created:
        nested = noun_path(citizen_created[1], depth + 1)
        return (nested[0], (*nested[1], "creator", "country of citizenship")) if nested else None
    # "the country where the sport associated with X was created/developed/originated"
    sport_origin = re.fullmatch(
        r"(?:the )?(?:country|nation) (?:that|which) (?:created|developed|produced|originated) "
        r"the sport (?:that is |which is )?(?:associated with|linked to|connected to|played by) (.+)",
        value,
        re.I,
    )
    if sport_origin:
        nested = noun_path(sport_origin[1], depth + 1)
        return (nested[0], (*nested[1], "sport", "country of origin")) if nested else None
    sport_origin_2 = re.fullmatch(
        r"(?:the )?(?:country|nation) where the sport (?:that is |which is )?"
        r"(?:associated with|linked to|connected to) (.+?) "
        r"(?:was created|was developed|originated|was originated)",
        value,
        re.I,
    )
    if sport_origin_2:
        nested = noun_path(sport_origin_2[1], depth + 1)
        return (nested[0], (*nested[1], "sport", "country of origin")) if nested else None
    # "the city where the founder of X operated/worked/lived"——X 本身可以带从句
    # （`the religion associated with Weingarten Abbey`），交给 `noun_path` 递归。
    founder_city = re.fullmatch(
        r"(?:the )?(?:city|town) (?:where|in which) the founder of (.+?) "
        r"(?:operated|worked|lived|was based|resided)",
        value,
        re.I,
    )
    if founder_city:
        nested = noun_path(founder_city[1], depth + 1)
        return (nested[0], (*nested[1], "founded by", "live")) if nested else None
    # "the city that houses the headquarters of the university that educated X"
    hq_university = re.fullmatch(
        r"(?:the )?(?:city|town) (?:that|which) (?:houses|hosts|is home to) the headquarters of "
        r"(?:the )?(?:university|college|school) (?:that|which) (?:educated|X) (.+)",
        value,
        re.I,
    )
    if hq_university is None:
        hq_university = re.fullmatch(
            r"(?:the )?(?:city|town) (?:that|which) (?:houses|hosts|is home to) the headquarters of "
            r"(?:the )?(?:university|college|school) (?:where|that) (.+?) (?:was educated|studied|"
            r"attended)",
            value,
            re.I,
        )
    if hq_university:
        nested = noun_path(hq_university[1], depth + 1)
        return (nested[0], (*nested[1], "educated at", "headquarters location")) if nested else None
    # "the city where the headquarters of the university that educated X is located"
    hq_university_2 = re.fullmatch(
        r"(?:the )?(?:city|town) (?:where|in which) the headquarters of (?:the )?"
        r"(?:university|college|school) (?:that|which) (?:educated|X) (.+?) "
        r"(?:is|are) located",
        value,
        re.I,
    )
    if hq_university_2:
        nested = noun_path(hq_university_2[1], depth + 1)
        return (nested[0], (*nested[1], "educated at", "headquarters location")) if nested else None
    # "the person whose ... "（`whose` 相对从句：取内层名字再沿关系走）
    whose = re.fullmatch(
        r"(?:the )?(person|man|woman|artist|author|creator) whose (.+?) is (.+)", value, re.I
    )
    if whose:
        nested = noun_path(whose[3], depth + 1)
        return (nested[0], (*nested[1], relation_key(whose[2]))) if nested else None
    # **所有格档案字段**（`Sarah Garcia's highest education level` /
    # `Barbara Jones's MBTI personality type`）：`<人名>'s <字段>` ⇒ 沿 `name` 找到人，
    # 再取该字段。字段名白名单与抽取侧同一张表（`grammar._PROFILE_FIELDS`）——
    # 不认的表就交回检索，不硬造关系名。
    possessive = re.fullmatch(r"(.+?)'s (.+)", literal)
    if possessive and re.fullmatch(r"[A-Z][A-Za-z'’-]+(?: [A-Z][A-Za-z'’-]+){0,3}", possessive[1]):
        # 字段名后面常跟时间限定（`… education level as of September 4, 2025`）——
        # 那是 `date_bounds` 的职责，字段名本身只取前半。
        field_text = re.split(r"\s+(?:as of|in|during|on)\s+", possessive[2], maxsplit=1)[0]
        # `latest / current / most recent` 只是**时间限定词**，不是字段名的一部分：
        # "X's latest job title" 仍是 `job title`（HaluMem 的通用问法，2026-10-19）。
        field_text = re.sub(r"^(?:latest|current|most recent|present)\s+", "", field_text, flags=re.I)
        field = relation_key(field_text)
        if field in _PROFILE_FIELD_NAMES:
            return possessive[1], (field,)
        # **所有格的通用关系**（`Gina's spouse` / `Jon's employer`）：字段名命中
        # `RELATION_NAMES` 的别名时，沿该关系走一步。⚠ 只认已声明的规范关系名，
        # 不认的字段照样交回检索。
        for relation, names in RELATION_NAMES.items():
            if field == relation_key(relation) or field in {relation_key(n) for n in names}:
                return possessive[1], (relation_key(relation),)
    # **`the <属性> of the country located, where/whose …`**（官方 MQuAKE 的逗号变体）：
    # 外层属性 + 国家嵌套，内层仍是 `noun_path` 的通用从句。
    if re.match(r"^(?:the )?(?:capital|continent|official language|head of state|"
                r"head of government|language|currency|population)\b", value, re.I):
        comma = re.fullmatch(
            r"(?:the )?(capital|continent|official language|head of state|head of government|"
            r"language|currency|population)(?: city| language)? (?:located,|located|of the country located,|"
            r"of the country located|of the country,|of the country) (?:where|whose|that|which) (.+)",
            value,
            re.I,
        )
        if comma:
            nested = noun_path(comma[2], depth + 1)
            if nested:
                relation = {
                    "capital": "capital",
                    "continent": "continent",
                    "official language": "official language",
                    "language": "official language",
                    "head of state": "head of state",
                    "head of government": "head of government",
                    "currency": "currency",
                    "population": "population",
                }[comma[1].lower()]
                return (nested[0], (*nested[1], "country of citizenship", relation))
    # **"the <属性> of the country located, where/whose …" 的逗号变体与后置从句**
    # （官方 MQuAKE，2026-10-19）：`In what city is the capital located, of the country …`。
    # 逗号两侧是同一个嵌套的两半，拼回去交给上面的通用规则。
    comma_tail = re.fullmatch(
        r"(?:the )?(capital|continent|official language|language|head of state|"
        r"head of government)(?: city)?(?: is)? (?:located,|located)(?: of)? (?:the )?(.+)",
        value,
        re.I,
    )
    if comma_tail:
        nested = noun_path(f"the {comma_tail[2]}", depth + 1)
        if nested:
            relation = {
                "capital": "capital",
                "continent": "continent",
                "official language": "official language",
                "language": "official language",
                "head of state": "head of state",
                "head of government": "head of government",
            }[comma_tail[1].lower()]
            return (nested[0], (*nested[1], relation))
    # **逗号插入语**（`In which city, the capital of a country, was X created`）：
    # 插入语描述的是**答案类型**（capital），不是路径的一步——只剥掉，不参与行走。
    if "," in value:
        stripped = re.sub(
            r",\s*(?:which is |being )?the (?:capital|continent|language) of a country,", ",", value
        )
        if stripped != value:
            return noun_path(stripped, depth + 1)
        # 逗号切开的两半是**同一个嵌套的两半**（"the capital located, of the country …"）：
        # 前一半是外层属性、后一半是内层从句——拼回去交给通用规则。
        head, _, tail = value.partition(",")
        if head.strip() and tail.strip():
            nested = noun_path(tail.strip(), depth + 1)
            if nested:
                return (nested[0], (*nested[1], relation_key(head.strip())))
    if not value or len(value) > 200 or re.search(r"\b(?:who|where|that|which)\b", value, re.I):
        return None
    return literal, ()


def question_path(text: str) -> tuple[str, tuple[str, ...]] | None:
    frames = (
        (None, r"(?:Who|What) is (.+)"),
        (None, r"What is the name of (.+)"),
        # "What city is the capital of …" / "What language is officially spoken in the country …"
        # （外层框架只剥掉疑问壳，内层仍是 noun_path 的嵌套关系；没有新增业务分支）。
        (None, r"What (?:city|town) (?:is|was) (.+)"),
        ("continent", r"What continent is (.+)"),
        ("continent", r"On which continent is (.+)"),
        (
            "languages spoken, written or signed",
            r"What language is (?:spoken|used) in (.+)",
        ),
        ("country of citizenship", r"What country is (.+?) a citizen of"),
        ("employee", r"(?:Who|What) does (.+?) work for"),
        ("employee", r"Where (?:does|do) (.+?) work"),
        ("educated at", r"Where was (.+?) educated"),
        (
            "educated at",
            r"(?:At which|What) university did (.+?) (?:receive their education|attend)",
        ),
        ("place of birth", r"Where was (.+?) born"),
        ("place of birth", r"(?:In which|In what|What) city was (.+?) born"),
        # 出生日期 / 居住地 / 教育程度 / MBTI 一类的**通用人物属性**（HaluMem 那批问法）。
        ("date of birth", r"(?:What|When) is (.+?)'s birth ?date"),
        ("date of birth", r"When was (.+?) born"),
        ("date of birth", r"(?:What|When) is the birth ?date of (.+)"),
        ("live", r"Where (?:does|do) (.+?) live(?: as of .+)?"),
        ("live", r"Where is (.+?) currently living"),
        (
            "notable work",
            r"What (?:notable achievement is|notable work is|is) (.+?) (?:best )?(?:known for|renowned for)",
        ),
        (None, r"Where is (.+)"),
        (None, r"(?:In what|What) city (?:can you find|houses) (.+)"),
        (None, r"What is the (?:location|name) of (.+)"),
        ("continent", r"Which continent (?:houses|contains) (.+)"),
        ("languages spoken, written or signed", r"(?:Which|What) language does (.+?) speak"),
        (
            "languages spoken, written or signed",
            r"(?:Which|What) language is the native tongue of (.+)",
        ),
        ("religion or worldview", r"To which religion is (.+?) connected"),
        ("religion or worldview", r"What faith is linked to (.+)"),
        ("place of death", r"Where did (.+?) (?:die|pass away)"),
        ("place of death", r"In (?:which|what) city did (.+?) (?:die|meet their demise)"),
        ("educated at", r"Where did (.+?) receive their education"),
        ("educated at", r"Which university educated (.+)"),
        ("employee", r"What organization employs (.+)"),
        ("occupation", r"In what field is (.+?) employed"),
        ("country of origin", r"(?:Which|What) country is credited with creating (.+)"),
        ("country of origin", r"In which country was (.+?) (?:created|originated)"),
        ("country of origin", r"What country is the birthplace of (.+)"),
        ("country of origin", r"Where was (.+?) (?:created|originated)"),
        ("genre", r"What (?:type|style) of music is (?:played by|associated with) (.+)"),
        ("genre", r"What genre of music does (.+?) play"),
        # "What type of music does the author of X play?"——`genre` 挂在**内层实体**上，
        # 不是最外层。内层仍由 `noun_path` 解析（`the author of X` ⇒ author 一步）。
        ("genre", r"What (?:type|style) of music does (.+?) play"),
        ("genre", r"Which genre of music is played by (.+)"),
        ("genre", r"Which genre of music does (.+?) play"),
        ("genre", r"What (?:type|kind) of music (?:is|was) (.+?) associated with"),
        (None, r"(?:Which|What) city (?:is|was) (.+)"),
        (None, r"(?:Which|What) city is home to (.+)"),
        (None, r"In (?:which|what) city is (.+?) located"),
        (None, r"Where is (.+?) located"),
        # **官方 MQuAKE 的疑问壳变体**（2026-10-19）：全部只是剥掉疑问词，
        # 内层仍是 `noun_path` 的通用嵌套——不新增业务分支。
        ("capital", r"(?:Which|What) city (?:serves as|acts as|is) the capital(?: city)? of (.+)"),
        (None, r"(?:Which|What) city (?:serves as|acts as) (.+)"),
        ("continent", r"In which continent is (.+)"),
        ("continent", r"Which continent contains (.+)"),
        ("continent", r"Which continent is (.+)"),
        ("continent", r"What continent contains (.+)"),
        ("continent", r"Which continent is (.+?) located in"),
        ("continent", r"Which continent is (.+?) situated"),
        ("continent", r"In which continent is (.+?) located"),
        ("continent", r"What continent is (.+?) located in"),
        ("continent", r"On which continent is (.+?) located"),
        ("country of origin", r"From which country does (.+) originate"),
        ("country of origin", r"Which country did (.+) originate from"),
        ("country of origin", r"Which country originated (.+)"),
        ("country of origin", r"What country originated (.+)"),
        ("country of origin", r"Which country is (.+?) from"),
        ("country of origin", r"In which country was (.+?) developed"),
        ("country of origin", r"Which country developed (.+)"),
        ("country of citizenship", r"In which country does (.+?) hold citizenship"),
        ("country of citizenship", r"Which country does (.+?) hold citizenship in"),
        ("country of citizenship", r"Which country is (.+?) a citizen of"),
        ("country of citizenship", r"In which country is (.+?) a citizen"),
        ("official language", r"Which language is officially spoken in (.+)"),
        ("official language", r"What language is officially spoken in (.+)"),
        ("official language", r"Which language is the official language of (.+)"),
        ("official language", r"What language is the official language of (.+)"),
        ("official language", r"Which language is used in (.+)"),
        ("official language", r"What language is used in (.+)"),
        ("official language", r"Which language is predominantly spoken in (.+)"),
        ("official language", r"What language is primarily spoken in (.+)"),
        ("official language", r"What is the primary language used in (.+)"),
        ("official language", r"What is the primary language of (.+)"),
        ("official language", r"Which language is the primary language of (.+)"),
        ("official language", r"In which language is (.+) written"),
        ("official language", r"In what language is (.+) written"),
        ("official language", r"In which language was (.+) written"),
        ("official language", r"In what language was (.+) written"),
        ("language of work or name", r"In which language was the work written that (.+)"),
        ("language of work or name", r"In what language was the work written that (.+)"),
        ("language of work or name", r"Which language was the work written in that (.+)"),
        ("language of work or name", r"In which language was the work written by (.+)"),
        ("sport", r"What sport is associated with (.+)"),
        ("sport", r"Which sport is associated with (.+)"),
        ("sport", r"What sport is related to (.+)"),
        ("sport", r"Which sport is related to (.+)"),
        ("sport", r"What sport is linked to (.+)"),
        ("sport", r"What is the sport that involves (.+)"),
        ("sport", r"Which sport involves (.+)"),
        ("sport", r"What sport does (.+?) play"),
        ("sport", r"Which sport does (.+?) play"),
        ("sport", r"What sport did (.+?) play"),
        ("position played on team / speciality", r"What position does (.+?) play"),
        ("position played on team / speciality", r"What position is played by (.+)"),
        ("notable work", r"What work is (.+?) renowned for"),
        ("notable work", r"What work is (.+?) famous for"),
        ("notable work", r"What is (.+?) renowned for"),
        ("notable work", r"What is (.+?) celebrated for"),
        ("notable work", r"What is (.+?) famous for"),
        ("place of death", r"Where did (.+?) die"),
        ("place of death", r"In which city did (.+?) die"),
        ("place of death", r"In what city did (.+?) die"),
        ("place of death", r"Which city did (.+?) die in"),
        ("place of death", r"What city did (.+?) die in"),
        ("place of birth", r"In which city was (.+?) born"),
        ("place of birth", r"In what city was (.+?) born"),
        ("place of birth", r"Which city was (.+?) born in"),
        ("place of birth", r"What city was (.+?) born in"),
        ("headquarters location", r"In which city is the headquarters of (.+?) located"),
        ("headquarters location", r"In what city is the headquarters of (.+?) located"),
        ("headquarters location", r"Which city houses the headquarters of (.+)"),
        ("headquarters location", r"What city is the headquarters of (.+?) located in"),
        ("headquarters location", r"Where is the headquarters of (.+?) located"),
        ("chief executive officer", r"Who is the CEO of (.+)"),
        ("chief executive officer", r"Who is the chief executive officer of (.+)"),
        ("spouse", r"To whom is (.+?) married"),
        ("spouse", r"Who is (.+?) married to"),
        ("spouse", r"Who is married to (.+)"),
        ("founded by", r"Who founded (.+)"),
        ("founded by", r"Who is the founder of (.+)"),
        ("founded by", r"What is the name of the founder of (.+)"),
        ("author", r"Who wrote (.+)"),
        ("author", r"Who is the author of (.+)"),
        ("creator", r"Who created (.+)"),
        ("creator", r"Who is the creator of (.+)"),
        ("director / manager", r"Who directed (.+)"),
        ("director / manager", r"Who is the director of (.+)"),
        ("performer", r"Who performed (.+)"),
        ("performer", r"Who is the performer of (.+)"),
        ("religion or worldview", r"What religion is associated with (.+)"),
        ("religion or worldview", r"Which religion is (.+?) affiliated with"),
        ("genre", r"What kind of music is associated with (.+)"),
        ("genre", r"What type of music is played by (.+)"),
        ("genre", r"What kind of music does (.+?) play"),
        ("genre", r"What type of music does (.+?) play"),
        ("capital", r"What is the capital city of (.+)"),
        ("capital", r"Which city is the capital of (.+)"),
        ("capital", r"What city is the capital of (.+)"),
        ("capital", r"What is the capital of (.+)"),
        ("educated at", r"Which university did (.+?) attend"),
        ("educated at", r"What university did (.+?) attend"),
        ("educated at", r"Where did (.+?) go to university"),
        ("occupation", r"What is the field of work of (.+)"),
        ("occupation", r"What field does (.+?) work in"),
        ("child", r"Who is the child of (.+)"),
        ("child", r"Who is the offspring of (.+)"),
        # 官方 MQuAKE 的"谓语在后"变体：`the creator of X best known for` /
        # `the country that X is a citizen of`。只剥壳，内层交给 `noun_path`。
        ("notable work", r"What is (.+?) best known for"),
        ("notable work", r"What is (.+?) known for"),
        ("country of citizenship", r"What is the language of (.+)"),
        ("official language", r"What is the language of the country that (.+?) is a citizen of"),
        ("official language", r"What is the language of the country where (.+?) is a citizen"),
        ("capital", r"What is the capital of the country that (.+?) is a citizen of"),
        ("capital", r"What is the capital of the country where (.+?) is a citizen"),
        ("capital", r"What is the capital of the country whose citizen is (.+)"),
        ("capital", r"What is the capital of the country whose citizen (.+)"),
        ("continent", r"What is the continent of the country that (.+?) is a citizen of"),
        ("continent", r"What is the continent of the country where (.+?) is a citizen"),
        ("place of death", r"What city is the place where (.+?) died"),
        ("place of birth", r"What city is the place where (.+?) was born"),
        ("place of birth", r"What city is the birthplace of (.+)"),
        ("place of death", r"What city is the death place of (.+)"),
        ("founded by", r"What is the name of the founder of the city that (.+)"),
        ("work location", r"In (?:which|what) city did (.+?) operate"),
        ("work location", r"In (?:which|what) city did (.+?) work"),
        ("work location", r"In (?:which|what) city does (.+?) work"),
        ("work location", r"What city did (.+?) work in"),
        ("work location", r"What city was the workplace of (.+)"),
        ("work location", r"In (?:which|what) city was the workplace of (.+)"),
        ("place of birth", r"Where was the founder of (.+?) born"),
        ("educated at", r"In (?:which|what) city is the headquarters of the university located, that (.+?)"),
        # **外层属性在疑问词之后、内层从句在 "located," 之后**（官方 MQuAKE 的逗号变体）：
        # 只剥壳；`the capital of …` / `the country where …` 的嵌套交给 `noun_path`。
        ("capital", r"In (?:which|what) city is the capital (?:located, of|located of|of) (.+)"),
        # "the continent of origin of the sport associated with X" / "the continent
        # houses the country of X"——外层属性 + 内层从句，仍是通用嵌套。
        (
            ("sport", "country of origin", "continent"),
            r"What is the continent of origin of the sport (?:that is |which is )?"
            r"(?:associated with|played by|linked to) (.+)",
        ),        ("continent", r"Which continent houses the country of (.+)"),
        ("continent", r"Which continent houses (.+)"),
        ("country of origin", r"Where does the sport associated with (.+?) originate"),
        ("country of origin", r"Where did the sport associated with (.+?) originate"),
        ("official language", r"What language is spoken in the country of the creator of (.+)"),
        ("official language", r"What language is spoken in the country of (.+)"),
        ("capital", r"In (?:which|what) city is the capital of the country located, where (.+)"),
        ("capital", r"In (?:which|what) city is the capital of the country located that (.+)"),
        ("continent", r"In (?:which|what) continent is the country located that (.+)"),
        ("continent", r"On (?:which|what) continent is the country located that (.+)"),
        ("continent", r"What continent is the country located in where (.+)"),
        ("continent", r"What continent is the country located in that (.+)"),
        (("place of birth", "continent"), r"On (?:which|what) continent is the city situated, where (.+?) was born"),
        ("country of origin", r"In (?:which|what) country was the sport developed, that (.+)"),
        ("country of origin", r"In (?:which|what) country was the sport developed that (.+)"),
        ("country of origin", r"In (?:which|what) country was the sport associated with (.+?) created"),
        # "In which language was the work written that X is renowned for?"——
        # 内层是 `the work that X is renowned for`（notable work），不是裸的人名：
        # 疑问壳里丢掉 "the work" 会让根变成 "X is renowned for"，永远走不通。
        ("language of work or name", r"In (?:which|what) language was the work written that (.+)"),
        ("language of work or name", r"What language was the work written in that (.+)"),
        ("language of work or name", r"In (?:which|what) language was the work written by (.+)"),
        ("headquarters location", r"Where is the headquarters of the university that (.+?) located"),
        (("educated at", "headquarters location"), r"Where is the headquarters of the university that (.+?) (?:attended|was educated)(?: at)?"),
        (("author", "educated at", "headquarters location"), r"Where is the headquarters of the university that the (?:author|creator|writer) of (.+?) attended"),
        ("headquarters location", r"Where is the headquarters of the university that (.+)"),
        (None, r"Where is (the headquarters of the university where .+?) located"),
        (None, r"Where is (the headquarters of the university where .+)"),
        ("headquarters location", r"In (?:which|what) city is the university's headquarters located, which is the alma mater of (.+)"),
        (None, r"In (?:which|what) city is (the headquarters of the university located, where .+?)"),
        ("headquarters location", r"In (?:which|what) city is the headquarters of (the university located, that .+?)"),
        ("continent", r"Which continent is home to the country where the sport associated with (.+?) was developed"),
        ("continent", r"Which continent is home to the country where (.+)"),
        ("continent", r"Which continent is home to the country that (.+)"),
        ("capital", r"Which city, serving as the capital, is located in the country where (.+)"),
        ("capital", r"Which city, serving as the capital, is located in the country that (.+)"),
        ("child", r"What is the name of the child of (.+)"),
        ("child", r"Who is the child of the person who founded (.+)"),
        ("child", r"Who created the offspring of (.+)"),
        ("child", r"Who created the child of (.+)"),
        ("genre", r"What (?:kind|type) of music is associated with the author of (.+)"),
        ("genre", r"What (?:kind|type) of music does the author of (.+?) play"),
        ("country of origin", r"In (?:which|what) city was (.+?) created"),
        ("country of citizenship", r"In (?:which|what) city,? is (.+?) a citizen"),
        # 插入语在疑问词之后（"In which city, being the capital of a country, is X …"）：
        # 插入语只说明答案类型，剥掉后内层仍是通用嵌套。
        (
            "country of citizenship",
            r"In (?:which|what) city, (?:being |which is )?the (?:capital|continent|language) "
            r"of a country, is (.+?) a citizen",
        ),
        ("country of origin", r"In (?:which|what) city, (?:being |which is )?the (?:capital|continent|language) of a country, was (.+?) created"),
        ("official language", r"What is the main language spoken in (.+)"),
        ("official language", r"Which language is mainly spoken in (.+)"),
        ("official language", r"What language is predominantly spoken in (.+)"),
        ("official language", r"What language is primarily used in (.+)"),
        ("official language", r"What language is mainly used in (.+)"),
        ("place of birth", r"Which city is where (the founder of .+?) was born"),
        ("place of birth", r"Which city is where (.+?) was born"),
        ("country of origin", r"What person founded the nation where (.+?) originated"),
        ("country of origin", r"Who founded the nation where (.+?) originated"),
        (None, r"Which city is home to (the headquarters of the university where .+)"),
        ("headquarters location", r"Which city is home to the headquarters of (.+)"),
        (
            ("country of citizenship", "official language"),
            r"What is the official language of the country of which (.+?) (?:is a citizen|holds citizenship)",
        ),
        (
            ("country of citizenship", "official language"),
            r"What is the official language spoken in the country where (.+?) (?:is a citizen|holds citizenship)",
        ),
        ("country of citizenship", r"What is the name of the director who worked with (.+)"),
        ("country of origin", r"What country (?:created|developed|originated) (.+)"),
        ("country of origin", r"Which country (?:created|developed|originated) (.+)"),
        ("country of origin", r"What country is the origin of (.+)"),
        ("genre", r"What genre of music is played by (.+)"),
        ("genre", r"What (?:type|kind) of music is played by (.+)"),
    )
    # Specific wrappers must precede every unconstrained wrapper: otherwise
    # "Where is ... located" binds "located" as part of the root entity.
    general = {
        r"(?:Who|What) is (.+)",
        r"Where is (.+)",
        r"(?:Which|What) city (?:is|was) (.+)",
        # 这些是**不带从句的裸壳**：必须排在"带 located in/that"的专用壳之后，
        # 否则 `What continent is the country located in where X` 会把整段从句当根
        # （2026-10-19 实测：root 变成 "… was created"）。
        r"(?:Which|What) continent is (.+)",
        r"(?:Which|What) continent contains (.+)",
        r"(?:Which|What) language is (.+)",
        r"(?:Which|What) sport is (.+)",
        r"(?:Which|What) position is (.+)",
        r"(?:Which|What) country is (.+)",
        r"(?:Which|What) city is (.+)",
        r"(?:Which|What) city is home to (.+)",
    }
    ordered = [frame for frame in frames if frame[1] not in general]
    ordered.extend(frame for frame in frames if frame[1] in general)
    for tail, pattern in ordered:
        m = re.fullmatch(pattern, text, re.I)
        if m:
            parsed = noun_path(m[1])
            if parsed:
                root, steps = parsed
                if pattern in {
                    r"What country is the birthplace of (.+)",
                    r"Where was (.+?) (?:created|originated)",
                } and (not steps or steps[-1] != "sport"):
                    continue
                if tail:
                    # tail 可以是**有序的几步**（`the continent of origin of the sport …`
                    # ⇒ sport → country of origin → continent），不是业务分支。
                    steps = (*steps, *(tail if isinstance(tail, tuple) else (tail,)))
                if (
                    pattern
                    in {
                        r"Where is (.+)",
                        r"(?:In what|What) city (?:can you find|houses) (.+)",
                        r"What is the (?:location|name) of (.+)",
                        r"(?:Which|What) city (?:is|was) (.+)",
                        r"(?:Which|What) city is home to (.+)",
                        r"In (?:which|what) city is (.+?) located",
                        r"Where is (.+?) located",
                    }
                    and not text.lower().startswith("what is the name")
                    and (
                        not steps
                        or steps[-1]
                        not in {
                            "headquarters location",
                            "place of birth",
                            "place of death",
                            "location of formation",
                        }
                    )
                ):
                    continue
                if steps:
                    return root, steps
    return None
