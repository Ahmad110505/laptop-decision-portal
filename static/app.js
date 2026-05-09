// static/app.js

document.addEventListener("DOMContentLoaded", function () {
    setupFormLoading();
    setupInputHints();
    setupNumberValidation();
    setupResultBadges();
    setupSmoothCards();
});


function setupFormLoading() {
    const forms = document.querySelectorAll("form");

    forms.forEach(function (form) {
        form.addEventListener("submit", function () {
            const submitButton = form.querySelector("button[type='submit']");

            if (submitButton) {
                submitButton.disabled = true;
                submitButton.innerText = "Processing...";
                submitButton.classList.add("loading");
            }
        });
    });
}


function setupInputHints() {
    const cpuScore = document.querySelector("input[name='cpu_score']");
    const gpuScore = document.querySelector("input[name='gpu_score']");
    const ramInput = document.querySelector("input[name='ram_gb']");
    const targetRamInput = document.querySelector("input[name='target_ram_gb']");
    const storageInput = document.querySelector("input[name='target_storage_gb']");

    if (cpuScore) {
        addHint(cpuScore, "Use a score from 1 to 10. Higher means stronger CPU.");
    }

    if (gpuScore) {
        addHint(gpuScore, "Use a score from 1 to 10. Integrated GPUs are usually low.");
    }

    if (ramInput) {
        addHint(ramInput, "Common values: 4, 8, 16, 32.");
    }

    if (targetRamInput) {
        addHint(targetRamInput, "Choose the RAM the user ideally needs.");
    }

    if (storageInput) {
        addHint(storageInput, "Common values: 256, 512, 1024.");
    }
}


function addHint(input, text) {
    const hint = document.createElement("small");
    hint.className = "input-hint";
    hint.innerText = text;

    input.insertAdjacentElement("afterend", hint);
}


function setupNumberValidation() {
    const numberInputs = document.querySelectorAll("input[type='number']");

    numberInputs.forEach(function (input) {
        input.addEventListener("input", function () {
            const value = Number(input.value);

            clearInputError(input);

            if (input.value === "") {
                return;
            }

            if (value < 0) {
                showInputError(input, "Value cannot be negative.");
            }

            if (
                (input.name === "cpu_score" || input.name === "gpu_score" || input.name === "target_gpu_score")
                && value > 10
            ) {
                showInputError(input, "Score should usually be between 1 and 10.");
            }

            if (input.name === "weight_kg" && value > 6) {
                showInputError(input, "This weight seems unusually high for a laptop.");
            }
        });
    });
}


function showInputError(input, message) {
    input.classList.add("input-error");

    let error = input.parentElement.querySelector(".field-error");

    if (!error) {
        error = document.createElement("small");
        error.className = "field-error";
        input.parentElement.appendChild(error);
    }

    error.innerText = message;
}


function clearInputError(input) {
    input.classList.remove("input-error");

    const error = input.parentElement.querySelector(".field-error");

    if (error) {
        error.remove();
    }
}


function setupResultBadges() {
    const cards = document.querySelectorAll(".card");

    cards.forEach(function (card) {
        const text = card.innerText.toLowerCase();

        if (text.includes("critical") || text.includes("high-risk") || text.includes("unusual")) {
            card.classList.add("card-danger");
        } else if (text.includes("medium") || text.includes("attention") || text.includes("slightly")) {
            card.classList.add("card-warning");
        } else if (text.includes("normal") || text.includes("low")) {
            card.classList.add("card-success");
        }
    });
}


function setupSmoothCards() {
    const cards = document.querySelectorAll(".card, .metric, .form-card, .result-hero");

    cards.forEach(function (card, index) {
        card.style.opacity = "0";
        card.style.transform = "translateY(12px)";

        setTimeout(function () {
            card.style.transition = "0.35s ease";
            card.style.opacity = "1";
            card.style.transform = "translateY(0)";
        }, 80 * index);
    });
}