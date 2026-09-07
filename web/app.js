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

// Абоненты взяты из присланного примера Биллинга (`data_example/ЛСФЛ.csv`),
// а не выдуманы: адреса, балансы и данные счётчиков настоящие для этого набора.
//
// ПАРА ПОДОБРАНА КОНТРАСТНОЙ, И ЭТО НЕ ОФОРМЛЕНИЕ. У первой отключение в
// графике есть, у второй его нет — без второй демонстрация показывала бы работу
// поиска, но не его способность ответить «отключений нет», а это разные вещи
// (то же соображение, что в `scripts/normalize_addresses.py`).
//
// Прежняя редакция держала здесь выдуманных абонентов с адресами «ул. Речная» и
// «пр. Заводской». Их нет в графике отключений: путь `outage` на стенде не
// показывался вовсе, сколько бы вопросов о воде ни задали.
const SUBSCRIBERS = [
  {
    id: "2100707718",
    name: "Михайлова Татьяна Алексеевна",
    account: "2100707718",
    // Есть в графике: Левобережный, 03.08 — 06.08.
    address: "г. Воронеж, ул. Саврасова, д. 2, кв. 9",
    balance: "−331,78 ₽",
    meter: "ХВС № 17078956, поверка до 01.06.2028",
  },
  {
    id: "2100303314",
    name: "Кузнецова Ольга Дмитриевна",
    account: "2100303314",
    // В графике отсутствует — проверяет ответ «плановых отключений нет».
    address: "г. Воронеж, ул. Курчатова, д. 50, кв. 12",
    balance: "4 553,89 ₽ (переплата)",
    meter: "ХВС № 21034512, поверка до 05.09.2030",
  },
];

const SESSION_ID = "sess-" + Math.random().toString(36).slice(2, 10);
const SYSTEM_PROMPT =
  "Ты помощник абонента водоканала. Отвечай кратко, по-русски, только по теме " +
  "жилищно-коммунальных услуг водоснабжения. Не выдумывай факты.";

const el = (id) => document.getElementById(id);
const ui = {
  subscriber: el("subscriber"),
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

/* --- профиль абонента ----------------------------------------------------- */

function currentSubscriber() {
  return SUBSCRIBERS.find((s) => s.id === ui.subscriber.value) || SUBSCRIBERS[0];
}

function renderProfile() {
  const s = currentSubscriber();
  const rows = [
    ["Лицевой счёт", s.account],
    ["Адрес", s.address],
    ["Баланс", s.balance],
    ["Счётчик", s.meter],
    ["Сессия", SESSION_ID],
  ];
  ui.profile.innerHTML = rows
    .map(([term, value]) => `<dt>${escape(term)}</dt><dd>${escape(value)}</dd>`)
    .join("");
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

function renderSources(node, sources) {
  if (!sources || !sources.length) return;
  const block = document.createElement("div");
  block.className = "sources";
  block.innerHTML =
    "<b>Источники:</b> " +
    sources.map((s) => escape(s.source_title || s.chunk_id)).join(" · ");
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
    // Этап 2 на прототипе не разворачивается: кнопка показывает устройство
    // пути записи, а не выполняет его.
    button.addEventListener("click", () => showConfirmation(action.label || action.action));
    block.appendChild(button);
  });
  node.parentElement.appendChild(block);
}

/* --- подтверждение операции записи (правило 4.7) --------------------------- */

function showConfirmation(what) {
  ui.confirmBody.textContent =
    'Будет выполнено действие: "' + what + '".\n' +
    "Ни одна операция записи не выполняется до этого подтверждения.";
  ui.confirm.hidden = false;
}

ui.confirmYes.addEventListener("click", () => {
  ui.confirm.hidden = true;
  addMessage("assistant", "Этап 2 на прототипе не разворачивается: запись не выполнялась. " +
    "Отказ и подтверждение в рабочем контуре одинаково попадают в аудит.");
});

ui.confirmNo.addEventListener("click", () => {
  ui.confirm.hidden = true;
  addMessage("assistant", "Отменено. В рабочем контуре отказ тоже записывается в аудит — " +
    "с пометкой, что запись не выполнялась.");
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

async function ask(question) {
  const subscriber = currentSubscriber();
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
          subscriber_id: subscriber.id,
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
          renderSources(node, event.data.sources);
          renderActions(node, event.data.suggested_actions);
          logEvent("metadata", (event.data.sources || []).length + " источник(ов)");
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
          // Провайдер не отдаёт расход в потоковом режиме; учёт появится на
          // шаге 10 вместе с телеметрией, а пока честнее показать прочерк,
          // чем ноль, который выглядит как «ничего не потрачено».
          ui.tokens.textContent = tokensUsed > 0 ? String(tokensUsed) : "— (учёт на шаге 10)";
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

ui.subscriber.innerHTML = SUBSCRIBERS.map(
  (s) => `<option value="${s.id}">${escape(s.name)}</option>`
).join("");
ui.subscriber.addEventListener("change", () => {
  renderProfile();
  addMessage("assistant", "Вошли как другой абонент. Сессия та же, идентификатор абонента " +
    "изменился — на этом и проверяется привязка сессии к абоненту.");
});

renderProfile();
