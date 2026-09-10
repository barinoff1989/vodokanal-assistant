/*
  Поведение демо-стенда.

  Компоненты соответствуют диаграмме C3 виджета: лента сообщений, потребитель
  потока, диалог подтверждения, кнопки предлагаемых действий, блок источников.
  Служебная панель на диаграмме отсутствует — она добавлена стенду, чтобы на
  защите было видно то, что абоненту не показывают.

  Поток разбирается вручную, а не через EventSource: тот умеет только GET, а
  контракт требует POST с телом запроса (раздел 7.1).
*/

"use strict";

// Справочник абонентов собирается из присланного примера Биллинга
// (`data_example/ЛСФЛ.csv` и др.) скриптом `scripts/build_subscriber_directory.py`
// в `web/subscribers.json`. Стенд читает его как статический файл — новый адрес
// в контракт API (правило 4.4) не добавляется.
//
// На рабочей системе этого справочника нет: `Session Context Provider` (C3
// виджета) берёт `subscriber_id` из SSO-сессии ЛК, искать некого. Поиск нужен
// только стенду — войти любым из примерных абонентов и показать разные сценарии
// (долг и переплата, отключение и его отсутствие), в том числе вручную
// воспроизвести проверки S1–S2: подтвердить черновик от чужого имени.
//
// Пара на случай, если файл не собран: без него стенд всё равно должен
// показывать оба ответа про отключения — «есть» и «нет».
const FALLBACK_SUBSCRIBERS = [
  {
    account: "2100202213",
    name: "Петров Николай Егорович",
    address: "г. Воронеж, Зои Космодемьянской, д. 50, кв. 78",
    balance: "−1 250,00 ₽",
    debt: "1 250,00 ₽",
    meters: "справочник не собран — запустите scripts/build_subscriber_directory.py",
  },
  {
    account: "2100303314",
    name: "Кузнецова Ольга Дмитриевна",
    address: "г. Воронеж, ул. Курчатова, д. 50, кв. 12",
    balance: "4 553,89 ₽",
    meters: "справочник не собран — запустите scripts/build_subscriber_directory.py",
  },
];

const DIRECTORY_LIMIT = 40; // сколько строк показывать разом — остальное сужается поиском

let directory = [];
let selected = null;

const SESSION_ID = "sess-" + Math.random().toString(36).slice(2, 10);
const SYSTEM_PROMPT =
  "Ты помощник абонента водоканала. Отвечай кратко, по-русски, только по теме " +
  "жилищно-коммунальных услуг водоснабжения. Не выдумывай факты.";

const el = (id) => document.getElementById(id);
const ui = {
  search: el("dir-search"),
  directory: el("directory"),
  dirMeta: el("dir-meta"),
  profile: el("profile"),
  feed: el("feed"),
  form: el("composer"),
  query: el("query"),
  send: el("send"),
  state: el("state"),
  counter: el("counter"),
  events: el("events"),
  clearEvents: el("clear-events"),
  confirm: el("confirm"),
  confirmBody: el("confirm-body"),
  confirmYes: el("confirm-yes"),
  confirmNo: el("confirm-no"),
  trace: el("m-trace"),
  pii: el("m-pii"),
  model: el("m-model"),
  finish: el("m-finish"),
  tokens: el("m-tokens"),
  ttft: el("m-ttft"),
};

let tokensUsed = 0;

/* --- справочник абонентов (роль Session Context Provider на стенде) -------- */

function currentSubscriber() {
  return selected || directory[0] || FALLBACK_SUBSCRIBERS[0];
}

// Поиск по фамилии или по номеру счёта. Строка из букв ищется в ФИО, строка из
// цифр — в номере: спрашивать «по чему искать» лишнее, а разнести можно по виду
// введённого.
function matches(entry, term) {
  if (!term) return true;
  const digits = term.replace(/\D/g, "");
  if (digits && /^\d/.test(term.trim())) return entry.account.includes(digits);
  return entry.name.toLowerCase().includes(term.toLowerCase());
}

function renderDirectory() {
  const term = ui.search.value.trim();
  const found = directory.filter((entry) => matches(entry, term));
  const shown = found.slice(0, DIRECTORY_LIMIT);

  ui.directory.innerHTML = shown
    .map((entry) => {
      const marks = [];
      if (entry.debt) marks.push('<span class="mark debt">долг</span>');
      if (entry.outage) marks.push('<span class="mark outage">отключение</span>');
      const current = selected && selected.account === entry.account ? " current" : "";
      return (
        `<li role="option" data-account="${escape(entry.account)}"` +
        ` class="dir-row${current}" tabindex="0">` +
        `<span class="dir-name">${escape(entry.name)}</span>` +
        `<span class="dir-acc">${escape(entry.account)}</span>` +
        (marks.length ? `<span class="dir-marks">${marks.join("")}</span>` : "") +
        "</li>"
      );
    })
    .join("");

  if (!directory.length) {
    ui.dirMeta.textContent = "справочник не собран: scripts/build_subscriber_directory.py";
  } else if (!found.length) {
    ui.dirMeta.textContent = "никто не найден";
  } else if (found.length > shown.length) {
    ui.dirMeta.textContent = `показаны первые ${shown.length} из ${found.length} — уточните поиск`;
  } else {
    ui.dirMeta.textContent = `найдено: ${found.length}`;
  }
}

function selectSubscriber(account, { announce } = { announce: true }) {
  const entry = directory.find((s) => s.account === account) ||
    FALLBACK_SUBSCRIBERS.find((s) => s.account === account);
  if (!entry) return;
  const changed = !selected || selected.account !== entry.account;
  selected = entry;
  renderDirectory();
  renderProfile();
  if (announce && changed) {
    addMessage(
      "assistant",
      "Вошли как " + entry.name + ". Сессия та же, идентификатор абонента " +
        "изменился — на этом и проверяется привязка сессии к абоненту.",
    );
  }
}

function renderProfile() {
  const s = currentSubscriber();
  const rows = [
    ["Абонент", s.name],
    ["Лицевой счёт", s.account],
    ["Адрес", s.address],
    ["Баланс", s.balance],
  ];
  if (s.debt) rows.push(["Задолженность", s.debt]);
  rows.push(["Счётчик", s.meters]);
  if (s.outage) rows.push(["Плановое отключение", s.outage]);
  if (s.phone) rows.push(["Телефон ЛК", s.phone]);
  rows.push(["Сессия", SESSION_ID]);
  ui.profile.innerHTML = rows
    .map(([term, value]) => `<dt>${escape(term)}</dt><dd>${escape(value)}</dd>`)
    .join("");
}

async function loadDirectory() {
  try {
    const response = await fetch("/static/subscribers.json", { cache: "no-store" });
    if (!response.ok) throw new Error("HTTP " + response.status);
    const payload = await response.json();
    directory = Array.isArray(payload.subscribers) ? payload.subscribers : [];
  } catch (error) {
    // Не пустой список: без справочника стенд всё равно должен что-то показывать.
    directory = FALLBACK_SUBSCRIBERS.slice();
    logEvent("error", "справочник абонентов не загружен: " + error.message);
  }
  selected = directory[0] || null;
  renderDirectory();
  renderProfile();
}

/* --- лента ---------------------------------------------------------------- */

function addMessage(role, text, extraClass) {
  const item = document.createElement("li");
  item.className = "msg " + role + (extraClass ? " " + extraClass : "");
  item.innerHTML =
    `<div class="who">${role === "user" ? "Вы" : "Помощник"}</div>` +
    `<div class="text"></div>`;
  item.querySelector(".text").textContent = text;
  ui.feed.appendChild(item);
  ui.feed.scrollTop = ui.feed.scrollHeight;
  return item.querySelector(".text");
}

function renderSources(node, sources, disclaimer) {
  if ((!sources || !sources.length) && !disclaimer) return;
  const block = document.createElement("div");
  block.className = "sources";

  if (sources && sources.length) {
    // Рядом с каждым источником — его близость и пометка происхождения.
    // Близость показана потому, что порог 0,872 подобран замером и на защите
    // спросят, почему найдено именно это; без числа ответить нечем.
    const items = sources.map((s) => {
      const title = escape(s.source_title || s.chunk_id);
      const score = typeof s.relevance_score === "number"
        ? ' <span class="score">' + s.relevance_score.toFixed(3) + "</span>"
        : "";
      // Ответ по документу, который написали мы, обязан быть отличим от ответа
      // по настоящему регламенту водоканала (ADR-012 предостерегает ровно от
      // смешения). Признак приходит с источником, а не выводится здесь заново.
      const mark = s.synthetic ? ' <span class="synthetic">демо-документ</span>' : "";
      return title + score + mark;
    });
    block.innerHTML = "<b>Источники:</b> " + items.join(" · ");
  }

  if (disclaimer) {
    const note = document.createElement("div");
    note.className = "disclaimer";
    note.textContent = disclaimer;
    block.appendChild(note);
  }

  node.parentElement.appendChild(block);
}

// Готовый бланк заявления — состояние Document Panel на C3 виджета. Приходит
// ссылкой в metadata-событии; сам лист (HTML под печать) отдаёт маршрут стенда
// /documents/<токен>, в контракт /v1 он не входит. Ссылка живёт ограниченный
// срок: в бланке ФИО и лицевой счёт абонента.
function renderDocument(node, url) {
  if (!url) return;
  const block = document.createElement("div");
  block.className = "document";
  const link = document.createElement("a");
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = "Открыть заполненный бланк для печати";
  block.appendChild(link);
  node.parentElement.appendChild(block);
}

function renderActions(node, actions) {
  if (!actions || !actions.length) return;
  const block = document.createElement("div");
  block.className = "actions";
  actions.forEach((action) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = action.label || action.action;
    button.addEventListener("click", () => chooseAction(action, block));
    block.appendChild(button);
  });
  node.parentElement.appendChild(block);
}

/*
  Что делает нажатие.

  Отказ уходит сразу: отменить черновик — действие безобидное, и переспрашивать
  «точно ли отменить» значит мешать.

  Подтверждение сначала открывает диалог с самим черновиком. Это и есть привал
  HITL из правила 4.7: случайное нажатие на подсказку под ответом не должно
  создавать обращение, а перед записью абонент обязан увидеть, что именно
  подтверждает.

  Кнопки после нажатия гаснут: черновик один, и второе нажатие относилось бы
  уже не к нему.
*/
function chooseAction(action, block) {
  block.querySelectorAll("button").forEach((button) => (button.disabled = true));
  if (action.action === "reject") {
    ask("Отменить", "reject");
    return;
  }
  showConfirmation();
}

/* --- подтверждение операции записи (правило 4.7) --------------------------- */

function showConfirmation() {
  // Черновик берётся из последнего ответа помощника: сервер показал его текстом,
  // и подтверждать абонент должен ровно то, что прочитал. Собрать текст здесь
  // заново значило бы показать одно, а отправить другое.
  const answers = ui.feed.querySelectorAll(".msg.assistant .text");
  const last = answers.length ? answers[answers.length - 1].textContent : "";
  const from = last.indexOf("Могу оформить обращение:");
  ui.confirmBody.textContent = from >= 0 ? last.slice(from) : last;
  ui.confirm.hidden = false;
}

ui.confirmYes.addEventListener("click", () => {
  ui.confirm.hidden = true;
  // Текст реплики нужен ленте диалога; решение принимает признак `intent`, а не
  // эти слова: правило 4.7 не признаёт подтверждением свободный текст.
  ask("Подтверждаю", "confirm");
});

ui.confirmNo.addEventListener("click", () => {
  ui.confirm.hidden = true;
  ask("Отменить", "reject");
});

/* --- служебная панель ------------------------------------------------------ */

function logEvent(kind, text) {
  const item = document.createElement("li");
  item.className = "ev-" + kind;
  item.textContent = kind + " · " + text;
  ui.events.appendChild(item);
  ui.events.scrollTop = ui.events.scrollHeight;
}

ui.clearEvents.addEventListener("click", () => {
  ui.events.innerHTML = "";
});

function setState(text, kind) {
  ui.state.textContent = text;
  ui.state.className = "state" + (kind ? " " + kind : "");
}

/* --- обращение к шлюзу ----------------------------------------------------- */

async function ask(question, intent) {
  const subscriber = currentSubscriber();
  if (intent) {
    // Нажатие кнопки — тоже реплика абонента, и в ленте она должна быть видна:
    // иначе ответ «Обращение зарегистрировано» выглядит взявшимся ниоткуда.
    addMessage("user", question);
  }
  const node = addMessage("assistant", "");
  node.classList.add("typing");

  const startedAt = performance.now();
  let firstChunkAt = null;

  setState("печатает…", "busy");
  ui.send.disabled = true;
  ui.trace.textContent = "—";
  ui.pii.textContent = "—";
  ui.model.textContent = "—";
  ui.finish.textContent = "—";
  ui.ttft.textContent = "—";

  try {
    const response = await fetch("/v1/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        system: SYSTEM_PROMPT,
        query: question,
        context: [],
        parameters: { stream: true },
        metadata: {
          // Намерение уходит признаком, а не словом в тексте: сервер по нему и
          // только по нему решает, выполнять ли запись (правило 4.7).
          ...(intent ? { intent: intent } : {}),
          // Номер лицевого счёта и есть идентификатор абонента: по нему
          // регистрация обращения ищет профиль в Биллинге
          // (`app/backend/registration.py`).
          subscriber_id: subscriber.account,
          session_id: SESSION_ID,
          channel: "lk_web",
          // Адрес идёт из профиля кабинета, а не из текста вопроса: ЛК его
          // знает, а «у нас третий день нет воды» адреса не содержит вовсе
          // (раздел 57.8). Без него ответчик отключений возвращает «не мой
          // случай», и вопрос уходит к модели — ровно то, что запрещает
          // ADR-013. Стенд адрес показывал, но не отправлял.
          address: subscriber.address,
        },
      }),
    });

    // Ошибка до начала потока приходит обычным ответом с телом RFC 7807.
    if (!response.ok && !response.headers.get("content-type")?.includes("event-stream")) {
      const problem = await response.json();
      node.parentElement.classList.add("problem");
      node.textContent = problem.detail || problem.title;
      node.classList.remove("typing");
      logEvent("error", problem.status + " " + problem.title);
      ui.trace.textContent = problem.trace_id || "—";
      setState("ошибка " + problem.status, "error");
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // События разделены пустой строкой; последний кусок может быть неполным.
      const parts = buffer.split("\n\n");
      buffer = parts.pop();

      for (const part of parts) {
        const event = parseEvent(part);
        if (!event) continue;

        if (event.name === "token") {
          if (firstChunkAt === null) {
            firstChunkAt = performance.now() - startedAt;
            ui.ttft.textContent = Math.round(firstChunkAt) + " мс";
          }
          node.textContent += event.data.delta || "";
          ui.feed.scrollTop = ui.feed.scrollHeight;
          logEvent("token", JSON.stringify(event.data.delta));
        } else if (event.name === "metadata") {
          renderSources(node, event.data.sources, event.data.disclaimer);
          renderActions(node, event.data.suggested_actions);
          renderDocument(node, event.data.document_url);
          logEvent(
            "metadata",
            (event.data.sources || []).length + " источник(ов)" +
              (event.data.document_url ? ", бланк по ссылке" : ""),
          );
        } else if (event.name === "done") {
          ui.trace.textContent = event.data.trace_id || "—";
          ui.finish.textContent = event.data.finish_reason || "—";
          if (event.data.finish_reason === "guardrail") {
            ui.finish.classList.add("alarm");
          }
          // Псевдоним провайдера против фактически ответившей модели: их
          // расхождение и есть срабатывание запасного провайдера.
          const routing = event.data.routing || {};
          if (routing.model) {
            // Имя модели у Яндекса — путь вида gpt://<каталог>/yandexgpt/latest;
            // осмысленны последние два звена, а не только хвост «latest».
            const parts = String(routing.model).split("/").filter(Boolean);
            const short = parts.slice(-2).join("/");
            const switched = routing.provider && !String(routing.model).includes(routing.provider);
            ui.model.textContent = switched ? short + " (запасной)" : short;
            if (switched) ui.model.classList.add("alarm");
          }
          const pii = event.data.pii_report || {};
          ui.pii.textContent = pii.pii_detected
            ? (pii.entities || []).join(", ")
            : "не найдено";
          if (pii.pii_detected) ui.pii.classList.add("alarm");
          const usage = event.data.usage || {};
          const spent = (usage.prompt_tokens || 0) + (usage.completion_tokens || 0);
          tokensUsed += spent;
          // Расход приходит в событии завершения. Прежде здесь стоял прочерк
          // с пометкой «учёт на шаге 10»: считалось, что провайдер не отдаёт
          // расход в потоке. Он отдаёт — если попросить `stream_options`
          // (проверено на YandexGPT и на локальной Qwen). Прочерк остаётся
          // только на случай, когда провайдер промолчал: ноль выглядел бы как
          // «ничего не потрачено».
          ui.tokens.textContent = tokensUsed > 0 ? String(tokensUsed) : "— (провайдер не прислал)";
          logEvent("done", event.data.finish_reason || "stop");
        } else if (event.name === "error") {
          node.parentElement.classList.add("problem");
          node.textContent = event.data.detail || event.data.title;
          ui.trace.textContent = event.data.trace_id || "—";
          logEvent("error", event.data.status + " " + event.data.title);
          setState("ошибка " + event.data.status, "error");
        }
      }
    }

    if (!node.parentElement.classList.contains("problem")) {
      setState("готов");
    }
  } catch (error) {
    node.parentElement.classList.add("problem");
    node.textContent = "Не удалось связаться с сервисом: " + error.message;
    setState("нет связи", "error");
    logEvent("error", String(error.message));
  } finally {
    node.classList.remove("typing");
    ui.send.disabled = false;
  }
}

function parseEvent(raw) {
  let name = "message";
  const dataLines = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) name = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  try {
    return { name, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return null;
  }
}

/* --- служебное ------------------------------------------------------------- */

function escape(value) {
  const node = document.createElement("span");
  node.textContent = String(value);
  return node.innerHTML;
}

ui.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = ui.query.value.trim();
  if (!question) return;
  addMessage("user", question);
  ui.query.value = "";
  ui.counter.textContent = "0 / 2000";
  ask(question);
});

ui.query.addEventListener("input", () => {
  ui.counter.textContent = ui.query.value.length + " / 2000";
});

// Отправка по Ctrl+Enter — привычно и не мешает многострочному вводу.
ui.query.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    ui.form.requestSubmit();
  }
});

ui.search.addEventListener("input", renderDirectory);

// Выбор строки — мышью или клавишей. Список длинный, и тянуться к мыши на каждый
// сценарий демонстрации мешало бы.
function pick(event) {
  const row = event.target.closest(".dir-row");
  if (!row) return;
  if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") return;
  event.preventDefault();
  selectSubscriber(row.dataset.account);
}
ui.directory.addEventListener("click", pick);
ui.directory.addEventListener("keydown", pick);

loadDirectory();
