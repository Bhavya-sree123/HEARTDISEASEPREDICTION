// ===============================
// Heart Disease Prediction
// script.js
// ===============================

const form = document.getElementById("predictionForm");
const loading = document.getElementById("loading");
const resultCard = document.getElementById("resultCard");
const predictionText = document.getElementById("predictionText");
const confidenceText = document.getElementById("confidenceText");
const adviceBox = document.getElementById("adviceBox");

// ===============================
// Form Submit
// ===============================

form.addEventListener("submit", async function (e) {

    e.preventDefault();

    loading.style.display = "block";
    resultCard.style.display = "none";

    // Collect form data
    const data = {

        age: Number(document.getElementById("age").value),
        sex: Number(document.getElementById("sex").value),
        cp: Number(document.getElementById("cp").value),
        trestbps: Number(document.getElementById("trestbps").value),
        chol: Number(document.getElementById("chol").value),
        fbs: Number(document.getElementById("fbs").value),
        restecg: Number(document.getElementById("restecg").value),
        thalach: Number(document.getElementById("thalach").value),
        exang: Number(document.getElementById("exang").value),
        oldpeak: Number(document.getElementById("oldpeak").value),
        slope: Number(document.getElementById("slope").value),
        ca: Number(document.getElementById("ca").value),
        thal: Number(document.getElementById("thal").value)

    };

    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {

                "Content-Type": "application/json"

            },

            body: JSON.stringify(data)

        });

        const result = await response.json();

        loading.style.display = "none";
        resultCard.style.display = "block";

        // ===============================
        // Prediction Result
        // ===============================

        if (result.prediction === 1) {

            predictionText.innerHTML =
                "❤️ Heart Disease Detected";

            predictionText.className =
                "text-center text-danger";

            adviceBox.className =
                "alert alert-danger";

            adviceBox.innerHTML =
                `<strong>Advice:</strong> ${result.advice}`;
        }

        else {

            predictionText.innerHTML =
                "💚 No Heart Disease Detected";

            predictionText.className =
                "text-center text-success";

            adviceBox.className =
                "alert alert-success";

            adviceBox.innerHTML =
                `<strong>Advice:</strong> ${result.advice}`;

        }

        // ===============================
        // Confidence
        // ===============================

        if (result.confidence !== undefined) {

            confidenceText.innerHTML =
                "Confidence : <strong>" +
                result.confidence.toFixed(2) +
                "%</strong>";

        }

        else {

            confidenceText.innerHTML = "";

        }

    }

    catch (error) {

        loading.style.display = "none";

        resultCard.style.display = "block";

        predictionText.innerHTML =
            "❌ Error";

        predictionText.className =
            "text-center text-danger";

        confidenceText.innerHTML = "";

        adviceBox.className =
            "alert alert-warning";

        adviceBox.innerHTML =
            "Unable to connect to the Flask server. Please make sure app.py is running.";

        console.error(error);

    }

});

// ===============================
// Reset Button
// ===============================

form.addEventListener("reset", function () {

    resultCard.style.display = "none";
    loading.style.display = "none";

});

// ===============================
// Input Animation
// ===============================

const inputs = document.querySelectorAll("input, select");

inputs.forEach(input => {

    input.addEventListener("focus", () => {

        input.style.transform = "scale(1.02)";

    });

    input.addEventListener("blur", () => {

        input.style.transform = "scale(1)";

    });

});

// ===============================
// End of File
// ===============================