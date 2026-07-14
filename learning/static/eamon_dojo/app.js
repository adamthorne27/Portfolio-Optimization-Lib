const lessonList = document.querySelector("#lesson-list");
const lessonStage = document.querySelector("#lesson-stage");
const lessonTitle = document.querySelector("#lesson-title");
const lessonSource = document.querySelector("#lesson-source");
const prereqList = document.querySelector("#prereq-list");
const problemText = document.querySelector("#problem-text");
const cellGoal = document.querySelector("#cell-goal");
const codeEditor = document.querySelector("#code-editor");
const output = document.querySelector("#output");
const runStatus = document.querySelector("#run-status");
const runButton = document.querySelector("#run-code");
const resetButton = document.querySelector("#reset-code");

let lessons = [];
let activeIndex = 0;

function setStatus(text, tone = "") {
  runStatus.textContent = text;
  runStatus.className = tone;
}

function renderLessonList() {
  lessonList.innerHTML = lessons
    .map(
      (lesson, index) => `
        <button class="lesson-link ${index === activeIndex ? "active" : ""}" data-index="${index}" type="button">
          <span>${lesson.stage}</span>
          ${lesson.title}
        </button>
      `,
    )
    .join("");
}

function renderLesson(index) {
  activeIndex = index;
  const lesson = lessons[activeIndex];
  lessonStage.textContent = lesson.stage;
  lessonTitle.textContent = lesson.title;
  lessonSource.textContent = lesson.source;
  prereqList.innerHTML = lesson.prerequisites.map((item) => `<li>${item}</li>`).join("");
  problemText.textContent = lesson.problem;
  cellGoal.textContent = "Implement the TODOs, then run this cell against the lesson tests.";
  codeEditor.value = lesson.starter_code;
  output.textContent = "";
  setStatus("Idle");
  renderLessonList();
}

async function loadLessons() {
  const response = await fetch("/api/lessons");
  if (!response.ok) throw new Error("Could not load lessons");
  const payload = await response.json();
  lessons = payload.lessons;
  renderLesson(0);
}

async function runCode() {
  const lesson = lessons[activeIndex];
  setStatus("Running");
  runButton.disabled = true;
  output.textContent = "";
  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        lesson_id: lesson.id,
        code: codeEditor.value,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Run failed");
    output.textContent = [result.stdout, result.stderr].filter(Boolean).join("\n");
    setStatus(result.ok ? "Passed" : "Failed", result.ok ? "pass" : "fail");
  } catch (error) {
    output.textContent = error.message;
    setStatus("Failed", "fail");
  } finally {
    runButton.disabled = false;
  }
}

lessonList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-index]");
  if (!button) return;
  renderLesson(Number(button.dataset.index));
});

resetButton.addEventListener("click", () => renderLesson(activeIndex));
runButton.addEventListener("click", runCode);

codeEditor.addEventListener("keydown", (event) => {
  if (event.key !== "Tab") return;
  event.preventDefault();
  const start = codeEditor.selectionStart;
  const end = codeEditor.selectionEnd;
  codeEditor.value = `${codeEditor.value.slice(0, start)}    ${codeEditor.value.slice(end)}`;
  codeEditor.selectionStart = start + 4;
  codeEditor.selectionEnd = start + 4;
});

loadLessons().catch((error) => {
  output.textContent = error.message;
  setStatus("Failed", "fail");
});
