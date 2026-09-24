"""Бланки заявлений абонента — образцы для самостоятельного заполнения.

ОТКУДА ВЗЯТА СТРУКТУРА. Не выдумана: состав полей сверен с публичными формами
водоканалов и с нормативной базой —

* Постановление Правительства РФ от 29.07.2013 № 644 (Правила холодного
  водоснабжения и водоотведения), п. 16 — перечень сведений в заявке на
  договор;
* Постановление Правительства РФ от 04.09.2013 № 776 (Правила организации
  учёта) и № 831 — ввод прибора учёта в эксплуатацию, поверка, снятие пломбы;
* публичные бланки РВК-Воронеж, Водоканала Нижнего Новгорода, Водоканала
  Санкт-Петербурга.

ЭТО ОБРАЗЦЫ, А НЕ ОФИЦИАЛЬНЫЕ ФОРМЫ. На прототипе настоящих утверждённых бланков
владельца нет (закрытая сеть, часть A8 запроса данных). Каждый образец помечен в
заголовке как ориентировочный — та же дисциплина, что у синтетических документов
базы знаний (`synthetic=True`) и у неутверждённого текста о качестве воды
(ADR-100): показать можно, выдавать за официальное — нельзя.

ТРИ РОДА ПОЛЕЙ.

* **Из Биллинга по лицевому счёту** — ФИО, счёт, адрес, дата, а в бланках про
  счётчик ещё назначение, заводской номер, показание, срок поверки (когда прибор
  один). Подставляются `TemplateResponder` без участия абонента.
* **Собираемые в диалоге** (`collect`) — то, чего в Биллинге нет, но абонент
  может назвать словами: удобная дата визита, способ поверки, вид справки.
  `TemplateResponder` спрашивает их по одному и вставляет ответы.
* **Заполняемые на бумаге** — прочерк `__________`: серия и номер паспорта,
  реквизиты правоустанавливающего документа. Спрашивать их в чате незачем,
  абонент впишет в распечатанный бланк.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from app.taxonomy import InquiryType

__all__ = [
    "BLANK",
    "FROM_INQUIRY_TYPE",
    "TEMPLATES",
    "TEMPLATE_MARKERS",
    "DocumentField",
    "DocumentTemplate",
    "is_template_request",
    "match_template",
    "render_template",
]

BLANK = "__________"
"""Незаполненное поле. Абонент вписывает от руки или в текстовом редакторе."""

_HEADER = (
    "ОБРАЗЕЦ ЗАЯВЛЕНИЯ (ориентировочный, не официальная форма)\n"
    "Уточните актуальный бланк в контакт-центре: {phone}\n"
    "{rule}\n"
    "{sep}\n"
)
_FOOTER = (
    "\n{sep}\n"
    'Кому: ООО "РВК-Воронеж"\n'
    "От: {full_name}\n"
    "Лицевой счёт: {account_number}\n"
    "Адрес: {address}\n"
    "Контактный телефон: {phone}\n"
    "\nДата: {date}          Подпись: __________ / {full_name} /\n"
)

_SEP = "—" * 60

# Поля, общие для всех бланков — из Биллинга и footer.
_BASE_FIELDS = (
    "full_name",
    "account_number",
    "address",
    "phone",
    "date",
    # Данные прибора учёта — подставляются, когда у лицевого счёта ровно один
    # счётчик. При двух и более остаются прочерком: какой из них имеет в виду
    # абонент, из вопроса неизвестно, а справочный блок под бланком приводит оба.
    "meter_kind",
    "meter_serial",
    "meter_reading",
    "meter_verify_by",
)

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


@dataclass(frozen=True, slots=True)
class DocumentField:
    """Поле, которое ассистент спрашивает у абонента при заполнении бланка."""

    name: str
    question: str
    """Готовый вопрос — без модели: спросить «когда вам удобно?» модель не нужна."""


@dataclass(frozen=True, slots=True)
class DocumentTemplate:
    """Один образец заявления."""

    template_id: str
    title: str
    rule: str
    """Правовое основание — печатается в шапке, чтобы образец было с чем сверить."""

    body: str
    """Текст заявления. Плейсхолдеры `{field}`: из Биллинга, из диалога или
    прочерк, если значения нет."""

    attachments: tuple[str, ...] = ()
    collect: tuple[DocumentField, ...] = ()
    """Поля, которые собираются у абонента по одному. Всё, чего здесь нет и что
    не пришло из Биллинга, остаётся прочерком для заполнения на бумаге."""

    def __post_init__(self) -> None:
        refs = set(_PLACEHOLDER.findall(self.body))
        known = set(_BASE_FIELDS) | {f.name for f in self.collect}
        unknown = refs - known
        if unknown:  # ловится тестом; здесь — чтобы опечатка не дошла до абонента
            raise ValueError(
                f"{self.template_id}: плейсхолдеры без источника: {sorted(unknown)}"
            )

    def field_names(self) -> set[str]:
        return set(_BASE_FIELDS) | {f.name for f in self.collect}

    def render(
        self, values: Mapping[str, str | None], *, company_phone: str = BLANK
    ) -> str:
        filled = {key: (values.get(key) or BLANK) for key in self.field_names()}
        parts = [
            _HEADER.format(rule=self.rule, sep=_SEP, phone=company_phone),
            f"ЗАЯВЛЕНИЕ\n{self.title}\n",
            self.body.format(**filled).strip(),
        ]
        if self.attachments:
            parts.append(
                "\nПриложения:\n"
                + "\n".join(f"  {i}. {name}" for i, name in enumerate(self.attachments, 1))
            )
        parts.append(_FOOTER.format(sep=_SEP, **filled))
        return "\n".join(parts)


_CONNECTION = DocumentTemplate(
    template_id="connection",
    title="о заключении договора холодного водоснабжения и водоотведения",
    rule="Постановление Правительства РФ от 29.07.2013 № 644, пункт 16",
    body=(
        "Прошу заключить договор холодного водоснабжения и водоотведения "
        "в отношении объекта по адресу: {address}.\n\n"
        "Сведения об объекте:\n"
        "  - вид объекта: {object_kind}\n"
        "  - площадь объекта, кв. м: {area}\n"
        "  - основание пользования: {ownership}\n"
        "  - реквизиты правоустанавливающего документа: __________\n\n"
        "Источник водоснабжения объекта: централизованная сеть.\n"
        "Наличие приборов учёта (да / нет; тип, заводской номер): __________\n"
        "Планируемая дата начала подачи воды: {start_date}\n\n"
        "Документ, удостоверяющий личность: серия __________ № __________, "
        "выдан __________ , дата выдачи __________ .\n"
        "Адрес регистрации по месту жительства: __________ ."
    ),
    attachments=(
        "копия документа, удостоверяющего личность",
        "копия правоустанавливающего документа на объект",
        "план расположения объекта с привязкой к местности (схема)",
        "доверенность — если заявление подаёт представитель",
    ),
    collect=(
        DocumentField(
            "object_kind",
            "Вид объекта — квартира, жилой дом или нежилое помещение?",
        ),
        DocumentField("area", "Площадь объекта в квадратных метрах?"),
        DocumentField(
            "ownership", "На каком основании пользуетесь объектом — собственность или наём?"
        ),
        DocumentField(
            "start_date", "С какой даты планируете начать пользоваться водой?"
        ),
    ),
)

_METER_VERIFICATION = DocumentTemplate(
    template_id="meter_verification",
    title="о проведении поверки прибора учёта холодной (горячей) воды",
    rule="Постановление Правительства РФ от 04.09.2013 № 776",
    body=(
        "Прошу организовать поверку прибора учёта по адресу: {address}.\n\n"
        "Прибор учёта:\n"
        "  - назначение (ХВС / ГВС): {meter_kind}\n"
        "  - заводской номер: {meter_serial}\n"
        "  - место установки: {install_place}\n"
        "  - показание на дату заявления: {meter_reading}\n"
        "  - дата истечения межповерочного интервала (по паспорту): {meter_verify_by}\n\n"
        "Удобная дата и время посещения: {visit_time}\n"
        "Способ поверки: {method}"
    ),
    attachments=(
        "копия паспорта прибора учёта",
        "копия акта ввода прибора учёта в эксплуатацию",
    ),
    collect=(
        DocumentField(
            "install_place",
            "Где установлен счётчик — например, под ванной, в туалете, в шкафу в коридоре?",
        ),
        DocumentField(
            "visit_time", "Когда вам удобно, чтобы пришёл специалист? Дата и время."
        ),
        DocumentField(
            "method",
            "Поверку сделать на месте без снятия счётчика или со снятием? "
            "Если не знаете — напишите «на месте».",
        ),
    ),
)

_METER_SEALING = DocumentTemplate(
    template_id="meter_sealing",
    title="о вводе прибора учёта в эксплуатацию (опломбировке)",
    rule="Постановление Правительства РФ от 04.09.2013 № 776, раздел IV",
    body=(
        "Прошу ввести в эксплуатацию (опломбировать) прибор учёта по адресу: "
        "{address}.\n\n"
        "Прибор учёта:\n"
        "  - назначение (ХВС / ГВС): {meter_kind}\n"
        "  - тип, марка: __________\n"
        "  - заводской номер: {meter_serial}\n"
        "  - место установки: {install_place}\n"
        "  - показание на момент установки: __________\n"
        "  - дата следующей поверки (по паспорту): {meter_verify_by}\n"
        "  - кто устанавливал прибор: {installer}\n\n"
        "Предлагаемая дата и время ввода в эксплуатацию: {commission_time}"
    ),
    attachments=(
        "копия паспорта прибора учёта",
        "копия документов о последней поверке (кроме новых приборов)",
        "договор и акт установки прибора учёта",
    ),
    collect=(
        DocumentField(
            "install_place", "Где установлен счётчик — под ванной, в туалете, в шкафу?"
        ),
        DocumentField(
            "installer",
            "Кто устанавливал счётчик — вы сами или организация? "
            "Если организация — напишите её название.",
        ),
        DocumentField(
            "commission_time", "Когда вам удобно провести опломбировку? Дата и время."
        ),
    ),
)

_METER_INSTALLATION = DocumentTemplate(
    template_id="meter_installation",
    title="об установке (замене) прибора учёта холодной (горячей) воды",
    rule="Федеральный закон от 23.11.2009 № 261-ФЗ; Правила № 776",
    body=(
        "Прошу согласовать установку (замену) прибора учёта по адресу: "
        "{address}.\n\n"
        "  - причина: {reason}\n"
        "  - назначение прибора (ХВС / ГВС): {meter_kind}\n"
        "  - предполагаемое место установки: {install_place}\n"
        "  - показание демонтируемого прибора (при замене): {meter_reading}\n"
        "  - заводской номер демонтируемого прибора (при замене): {meter_serial}\n\n"
        "Работы выполняет: {performer}\n"
        "Желаемый срок выполнения: {deadline}"
    ),
    attachments=(
        "копия документа, удостоверяющего личность",
        "копия правоустанавливающего документа на объект",
    ),
    collect=(
        DocumentField(
            "reason",
            "Причина — первичная установка, замена по сроку поверки или прибор вышел из строя?",
        ),
        DocumentField(
            "install_place", "Где стоит (будет стоять) счётчик?"
        ),
        DocumentField(
            "performer", "Работы выполните сами или подрядная организация?"
        ),
        DocumentField("deadline", "В какой срок хотите выполнить?"),
    ),
)

_METER_UNSEALING = DocumentTemplate(
    template_id="meter_unsealing",
    title="о снятии пломбы с прибора учёта",
    rule="Правила № 776, пункт 49 (уведомление не менее чем за 2 рабочих дня)",
    body=(
        "Прошу снять пломбу с прибора учёта по адресу: {address} "
        "для последующей {purpose}.\n\n"
        "Прибор учёта:\n"
        "  - назначение (ХВС / ГВС): {meter_kind}\n"
        "  - заводской номер: {meter_serial}\n"
        "  - показание на дату заявления: {meter_reading}\n\n"
        "Предполагаемая дата работ: {work_date} "
        "(не ранее чем через 2 рабочих дня после подачи заявления).\n"
        "Контактное лицо на объекте и телефон: {contact}"
    ),
    collect=(
        DocumentField(
            "purpose", "Для чего снять пломбу — поверка, замена или ремонт счётчика?"
        ),
        DocumentField(
            "work_date",
            "На какую дату планируете работы? Не раньше чем через 2 рабочих дня.",
        ),
        DocumentField(
            "contact", "Кто будет на объекте и по какому телефону с ним связаться?"
        ),
    ),
)

_CERTIFICATE = DocumentTemplate(
    template_id="certificate",
    title="о выдаче справки о состоянии расчётов (задолженности)",
    rule="Правила № 644",
    body=(
        "Прошу выдать справку о состоянии расчётов по лицевому счёту "
        "{account_number} (адрес: {address}).\n\n"
        "  - вид справки: {cert_kind}\n"
        "  - справка требуется по состоянию на дату: {as_of_date}\n"
        "  - цель получения: {purpose}\n"
        "  - способ получения: {delivery}\n\n"
        "Показания приборов учёта на дату заявления переданы: __________ (да / нет)."
    ),
    attachments=(
        "копия документа, удостоверяющего личность",
        "копия правоустанавливающего документа — при подготовке к сделке",
    ),
    collect=(
        DocumentField(
            "cert_kind",
            "Какая справка нужна — об отсутствии задолженности, о наличии "
            "или о состоянии расчётов на дату?",
        ),
        DocumentField(
            "purpose",
            "Для чего нужна справка — для сделки купли-продажи, для суда, "
            "для приставов или другое?",
        ),
        DocumentField(
            "as_of_date", "На какую дату нужна справка? Можно «на сегодня»."
        ),
        DocumentField(
            "delivery", "Как получить справку — лично, по электронной почте или почтой?"
        ),
    ),
)

_INSPECTION = DocumentTemplate(
    template_id="inspection",
    title="о проведении обследования (узла учёта, ввода, сетей)",
    rule="Правила № 776; Правила № 644",
    body=(
        "Прошу провести обследование по адресу: {address}.\n\n"
        "  - что обследовать: {what}\n"
        "  - причина обращения: {reason}\n"
        "  - выявленные признаки (при наличии): __________\n\n"
        "Удобная дата и время посещения: {visit_time}\n"
        "Контактное лицо на объекте и телефон: {contact}"
    ),
    collect=(
        DocumentField(
            "what",
            "Что обследовать — узел учёта, водопроводный ввод, канализационный "
            "выпуск или место присоединения?",
        ),
        DocumentField("reason", "Опишите причину обращения."),
        DocumentField("visit_time", "Когда вам удобно принять специалиста?"),
        DocumentField(
            "contact", "Кто будет на объекте и по какому телефону с ним связаться?"
        ),
    ),
)


TEMPLATES: dict[str, DocumentTemplate] = {
    t.template_id: t
    for t in (
        _CONNECTION,
        _METER_VERIFICATION,
        _METER_SEALING,
        _METER_INSTALLATION,
        _METER_UNSEALING,
        _CERTIFICATE,
        _INSPECTION,
    )
}


FROM_INQUIRY_TYPE: dict[InquiryType, str] = {
    InquiryType.METER_VERIFICATION: "meter_verification",
    InquiryType.METER_SEALING: "meter_sealing",
    InquiryType.METER_INSTALLATION: "meter_installation",
    InquiryType.METER_UNSEALING: "meter_unsealing",
    InquiryType.CERTIFICATE: "certificate",
    InquiryType.INSPECTION: "inspection",
}
"""Тип обращения -> образец. `connection` сюда не входит: заключение договора на
регистрационной оси отдельным типом не выделено (ADR-300)."""


# Слова, по которым образец узнаётся в вопросе абонента. Порядок проверки — от
# частного к общему, первое совпадение выигрывает.
TEMPLATE_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("meter_verification", ("поверк",)),
    ("meter_sealing", ("опломб", "ввод в эксплуат", "ввод прибора", "ввод счёт", "ввод счет")),
    ("meter_unsealing", ("снятие пломб", "снять пломб", "распломб")),
    ("meter_installation", ("установк", "замен", "монтаж прибор")),
    ("certificate", ("справк",)),
    ("inspection", ("обследован",)),
    (
        "connection",
        ("подключен", "присоединен", "заключить договор", "заключени договор", "новый договор"),
    ),
)

_REQUEST_MARKERS: tuple[str, ...] = (
    "шаблон",
    "бланк",
    "образец",
    "форма заявлен",
    "форму заявлен",
    "как написать заявлен",
    "как составить заявлен",
    "пример заявлен",
    "распечат",
)


def is_template_request(lowered: str) -> bool:
    """Просит ли абонент сам бланк, а не оформление обращения ассистентом."""
    return any(marker in lowered for marker in _REQUEST_MARKERS)


def match_template(lowered: str) -> str | None:
    """Какой образец подходит вопросу. `None` — не разобрали, какой именно."""
    for template_id, markers in TEMPLATE_MARKERS:
        if any(marker in lowered for marker in markers):
            return template_id
    return None


def render_template(
    template_id: str,
    *,
    prefill: Mapping[str, str | None] | None = None,
    company_phone: str = BLANK,
) -> str:
    """Отрисовать образец, подставив известные данные абонента.

    :raises KeyError: неизвестный `template_id`.
    """
    return TEMPLATES[template_id].render(dict(prefill or {}), company_phone=company_phone)
