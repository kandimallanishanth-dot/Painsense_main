// =========================================================
// PAINSENSE DASHBOARD
// Live Flask API + History
// =========================================================

let painScores = [];
let timestamps = [];
let chart = null;


// =========================================================
// UPDATE PAIN SCORE
// =========================================================

function updatePainScore(score) {

    const element =
        document.getElementById("painScore");

    const status =
        document.getElementById("painStatus");

    if (
        score === null ||
        score === undefined
    ) {

        element.textContent = "—";

        status.textContent =
            "Waiting for prediction";

        return;
    }

    score = Number(score);

    element.textContent =
        score.toFixed(1);

    if (score >= 6) {

        status.textContent =
            "High pain detected";

    }
    else if (score >= 3) {

        status.textContent =
            "Moderate pain";

    }
    else {

        status.textContent =
            "Low / No pain";
    }
}


// =========================================================
// UPDATE PREDICTION
// =========================================================

function updatePrediction(prediction) {

    const element =
        document.getElementById("prediction");

    if (!prediction) {

        element.textContent = "—";

        return;
    }

    element.textContent =
        prediction;
}


// =========================================================
// UPDATE PAIN PROBABILITY
// =========================================================

function updatePainProbability(score) {

    const element =
        document.getElementById("painProbability");

    if (
        score === null ||
        score === undefined
    ) {

        element.textContent = "—";

        return;
    }

    element.textContent =
        Number(score).toFixed(1) + " / 10";
}


// =========================================================
// UPDATE SEQUENCE
// =========================================================

function updateSequence(
    current,
    total
) {

    const element =
        document.getElementById("sequenceStatus");

    element.textContent =
        `${current} / ${total}`;
}


// =========================================================
// UPDATE ALERT
// =========================================================

function updateAlert(
    alert,
    reason
) {

    const alertElement =
        document.getElementById("alertStatus");

    const reasonElement =
        document.getElementById("alertReason");


    if (alert) {

        alertElement.textContent =
            "ALERT";

        alertElement.style.color =
            "#c62828";

        reasonElement.textContent =
            reason ||
            "Pain alert triggered";

    }
    else {

        alertElement.textContent =
            "STANDBY";

        alertElement.style.color =
            "#16803c";

        reasonElement.textContent =
            "No active alert";
    }
}


// =========================================================
// CREATE PAIN PROBABILITY CHART
// =========================================================

function createChart() {

    const canvas =
        document.getElementById("painChart");

    if (!canvas) {
        return;
    }

    const context =
        canvas.getContext("2d");

    chart = new Chart(
        context,
        {

            type: "line",

            data: {

                labels: timestamps,

                datasets: [

                    {
                        label: "Pain score",

                        data: painScores,

                        tension: 0.35,

                        borderWidth: 2,

                        pointRadius: 3,

                        fill: false
                    }

                ]
            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                scales: {

                    y: {

                        min: 0,

                        max: 10,

                        title: {

                            display: true,

                            text: "Pain score"
                        }
                    },

                    x: {

                        title: {

                            display: true,

                            text: "Time"
                        }
                    }
                },

                plugins: {

                    legend: {

                        display: false
                    }
                }
            }
        }
    );
}


// =========================================================
// ADD PAIN PROBABILITY TO CHART
// =========================================================

function addPainScore(
    probability,
    time
) {

    if (
        probability === null ||
        probability === undefined
    ) {
        return;
    }


    painScores.push(
        Number(probability)
    );


    timestamps.push(
        time ||
        new Date().toLocaleTimeString()
    );


    // Keep latest 10 observations.

    if (painScores.length > 10) {

        painScores.shift();

        timestamps.shift();
    }


    if (chart) {

        chart.data.labels =
            timestamps;

        chart.data.datasets[0].data =
            painScores;


        // Dynamically adjust Y-axis.

        updateChartScale();


        chart.update();
    }
}


// =========================================================
// UPDATE CHART SCALE
// =========================================================

function updateChartScale() {

    if (
        !chart ||
        painScores.length === 0
    ) {
        return;
    }


    chart.options.scales.y.min = 0;
    chart.options.scales.y.max = 10;
}


// =========================================================
// UPDATE RECENT OBSERVATIONS TABLE
// =========================================================

function updateHistory(history) {

    if (!Array.isArray(history)) {

        console.warn(
            "Invalid history data:",
            history
        );

        return;
    }


    // Find Recent Observations table.

    const tableBody =
        document.querySelector(
            "table tbody"
        );


    if (!tableBody) {

        console.warn(
            "Recent Observations table body not found."
        );

        return;
    }


    // Clear current table.

    tableBody.innerHTML = "";


    // No history.

    if (history.length === 0) {

        const row =
            document.createElement("tr");

        row.innerHTML = `
            <td
                colspan="4"
                style="
                    text-align:center;
                    padding:24px;
                "
            >
                No observations recorded yet.
            </td>
        `;

        tableBody.appendChild(row);

        return;
    }


    // Newest observations first.

    const recent =
        history
            .slice()
            .reverse()
            .slice(0, 10);


    recent.forEach(
        function (item) {

            const row =
                document.createElement("tr");


            // -----------------------------------------
            // TIME
            // -----------------------------------------

            let timeText = "—";


            if (
                item.timestamp !== null &&
                item.timestamp !== undefined
            ) {

                timeText =
                    new Date(
                        Number(item.timestamp) * 1000
                    ).toLocaleTimeString();
            }


            // -----------------------------------------
            // PAIN PROBABILITY
            // -----------------------------------------

            let probabilityText = "—";

            const historyScore =
                item.score !== null &&
                item.score !== undefined
                    ? item.score
                    : item.probability;


            if (
                historyScore !== null &&
                historyScore !== undefined
            ) {

                probabilityText =
                    Number(historyScore).toFixed(1);
            }


            // -----------------------------------------
            // PREDICTION
            // -----------------------------------------

            const prediction =
                item.prediction ||
                "—";


            // -----------------------------------------
            // ALERT
            // -----------------------------------------

            const alertText =
                item.alert
                    ? "ALERT"
                    : "—";


            // -----------------------------------------
            // CREATE TABLE ROW
            // -----------------------------------------

            row.innerHTML = `
                <td>${timeText}</td>

                <td>
                    ${probabilityText}
                </td>

                <td>
                    ${prediction}
                </td>

                <td>
                    ${alertText}
                </td>
            `;


            tableBody.appendChild(row);
        }
    );
}


// =========================================================
// GET LIVE DATA FROM FLASK
// =========================================================

async function fetchState() {

    try {

        const response =
            await fetch("/api/state");


        if (!response.ok) {

            throw new Error(
                "State API request failed"
            );
        }


        const data =
            await response.json();


        console.log(
            "PainSense state:",
            data
        );


        // -----------------------------------------
        // LIVE DASHBOARD VALUES
        // -----------------------------------------

        updatePainScore(
            data.score
        );


        updatePrediction(
            data.prediction
        );


        updatePainProbability(
            data.score
        );


        updateSequence(
            data.sequence_current,
            data.sequence_total
        );


        updateAlert(
            data.alert,
            data.alert_reason
        );

    }
    catch (error) {

        console.error(
            "PainSense state API error:",
            error
        );
    }
}


// =========================================================
// GET HISTORY FROM FLASK
// =========================================================

async function fetchHistory() {

    try {

        const response =
            await fetch("/api/history");


        if (!response.ok) {

            throw new Error(
                "History API request failed"
            );
        }


        const history =
            await response.json();


        console.log(
            "PainSense history:",
            history
        );


        // =============================================
        // UPDATE RECENT OBSERVATIONS TABLE
        // =============================================

        updateHistory(
            history
        );


        // =============================================
        // REBUILD CHART FROM DATABASE HISTORY
        // =============================================

        painScores = [];

        timestamps = [];


        // Take latest 10 records.

        const recent =
            history
                .slice(-10);


        recent.forEach(
            function (item) {

                const historyScore =
                    item.score !== null &&
                    item.score !== undefined
                        ? item.score
                        : item.probability;

                if (
                    historyScore === null ||
                    historyScore === undefined
                ) {

                    return;
                }

                painScores.push(
                    Number(historyScore)
                );


                // -----------------------------------------
                // CONVERT TIMESTAMP
                // -----------------------------------------

                let timeText =
                    "—";


                if (
                    item.timestamp !== null &&
                    item.timestamp !== undefined
                ) {

                    timeText =
                        new Date(
                            Number(item.timestamp) * 1000
                        ).toLocaleTimeString();
                }


                timestamps.push(
                    timeText
                );
            }
        );


        // =============================================
        // UPDATE CHART
        // =============================================

        if (chart) {

            chart.data.labels =
                timestamps;

            chart.data.datasets[0].data =
                painScores;


            // Automatically choose an appropriate
            // Y-axis range.

            updateChartScale();


            chart.update();
        }

    }
    catch (error) {

        console.error(
            "PainSense history API error:",
            error
        );
    }
}


// =========================================================
// START DASHBOARD
// =========================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        console.log(
            "PainSense dashboard started."
        );


        // Create empty chart.

        createChart();


        // Get live state immediately.

        fetchState();


        // Get history immediately.

        fetchHistory();


        // Update live state every second.

        setInterval(
            fetchState,
            1000
        );


        // Update history/chart every two seconds.

        setInterval(
            fetchHistory,
            2000
        );

    }
);