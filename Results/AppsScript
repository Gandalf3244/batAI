/**
 * Consolidates the Raw sheet by date:
 *  - Sums vocalization totals + duration (used only for recalculating per-hour rates)
 *  - Outputs ONLY the per-hour vocalization columns (no raw totals)
 *  - Carries feeding columns as-is (same value for every row sharing a date)
 * Results are written to a sheet named "Consolidated".
 */
function consolidateByDate() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();

  const rawSheet = ss.getSheetByName("Raw") || ss.getSheets()[0];
  const rawData  = rawSheet.getDataRange().getValues();
  const headers  = rawData[0];

  const idx = {};
  headers.forEach((h, i) => { idx[h] = i; });

  const sumCols = [
    "Rods_Fighting_Total_Calls",
    "Rods_Talking_Total_Calls",
    "Straws_Fighting_Total_Calls",
    "Straws_Fighting_Talking_Total_Calls",
    "Straws_Talking_Total_Calls",
    "Straws_Want_Food_Total_Calls",
    "Rods_Total_Vocalizations",
    "Straws_Total_Vocalizations",
    "Total Vocalizations",
    "Total_Duration_Minutes",
    "Total_Duration_Hours"
  ];

  const perHourPairs = [
    ["Rods_Fighting_Calls_Per_Hour",           "Rods_Fighting_Total_Calls"],
    ["Rods_Talking_Calls_Per_Hour",            "Rods_Talking_Total_Calls"],
    ["Straws_Fighting_Calls_Per_Hour",         "Straws_Fighting_Total_Calls"],
    ["Straws_Fighting_Talking_Calls_Per_Hour", "Straws_Fighting_Talking_Total_Calls"],
    ["Straws_Talking_Calls_Per_Hour",          "Straws_Talking_Total_Calls"],
    ["Straws_Want_Food_Calls_Per_Hour",        "Straws_Want_Food_Total_Calls"],
    ["Rods_Total_Calls_Per_Hour",              "Rods_Total_Vocalizations"],
    ["Straws_Total_Calls_Per_Hour",            "Straws_Total_Vocalizations"],
    ["Combined_Total_Calls_Per_Hour",          "Total Vocalizations"]
  ];

  const feedingCols = [
    "Next-Day Food Offered",
    "Letter",
    "amount eaten",
    "Adjusted consumption"
  ];

  const outHeaders = [
    "Date",
    "Total_Duration_Hours",
    "Rods_Fighting_Calls_Per_Hour",
    "Rods_Talking_Calls_Per_Hour",
    "Straws_Fighting_Calls_Per_Hour",
    "Straws_Fighting_Talking_Calls_Per_Hour",
    "Straws_Talking_Calls_Per_Hour",
    "Straws_Want_Food_Calls_Per_Hour",
    "Rods_Total_Calls_Per_Hour",
    "Straws_Total_Calls_Per_Hour",
    "Combined_Total_Calls_Per_Hour",
    "Next-Day Food Offered",
    "Letter",
    "amount eaten",
    "Adjusted consumption"
  ];

  const dateOrder = [];
  const agg       = {};

  for (let r = 1; r < rawData.length; r++) {
    const row     = rawData[r];
    const dateVal = row[idx["Date"]];

    // Skip rows with no date (blank rows, zero rows, etc.)
    if (!dateVal || dateVal === "" || dateVal === 0) continue;

    const dateKey = dateVal instanceof Date
                    ? Utilities.formatDate(dateVal, ss.getSpreadsheetTimeZone(), "M/d/yyyy")
                    : String(dateVal);

    if (!agg[dateKey]) {
      dateOrder.push(dateKey);
      agg[dateKey] = { dateVal };
      sumCols.forEach(c => { agg[dateKey][c] = 0; });
      feedingCols.forEach(c => { agg[dateKey][c] = ""; });
    }

    sumCols.forEach(c => {
      const v = parseFloat(row[idx[c]]);
      if (!isNaN(v)) agg[dateKey][c] += v;
    });

    feedingCols.forEach(c => {
      if (agg[dateKey][c] === "" && row[idx[c]] !== "") {
        agg[dateKey][c] = row[idx[c]];
      }
    });
  }

  dateOrder.forEach(dateKey => {
    const a   = agg[dateKey];
    const hrs = a["Total_Duration_Hours"] || 0;
    perHourPairs.forEach(([phCol, totCol]) => {
      a[phCol] = hrs > 0 ? parseFloat((a[totCol] / hrs).toFixed(1)) : 0;
    });
  });

  const outRows = [outHeaders];
  dateOrder.forEach(dateKey => {
    const a   = agg[dateKey];
    const row = outHeaders.map(h => {
      if (h === "Date") return a.dateVal;
      if (a[h] !== undefined) return a[h];
      return "";
    });
    outRows.push(row);
  });

  let outSheet = ss.getSheetByName("Consolidated");
  if (outSheet) {
    outSheet.clearContents();
  } else {
    outSheet = ss.insertSheet("Consolidated");
  }

  outSheet.getRange(1, 1, outRows.length, outRows[0].length).setValues(outRows);
  outSheet.getRange(1, 1, 1, outHeaders.length).setFontWeight("bold");
  outSheet.autoResizeColumns(1, outHeaders.length);

  SpreadsheetApp.getUi().alert(
    "Done! " + (outRows.length - 1) + " consolidated rows written to the 'Consolidated' sheet."
  );
}

// ── Outlier detection: IQR method ─────────────────────────────────────────────
function iqrBounds(arr) {
  const sorted = arr.slice().sort((a, b) => a - b);
  const n = sorted.length;
  const q1 = sorted[Math.floor(n * 0.25)];
  const q3 = sorted[Math.floor(n * 0.75)];
  const iqr = q3 - q1;
  return { lower: q1 - 1.5 * iqr, upper: q3 + 1.5 * iqr };
}

function removeOutliers(xArr, yArr) {
  const xBounds = iqrBounds(xArr);
  const yBounds = iqrBounds(yArr);

  const xClean = [], yClean = [];
  let nRemoved = 0;

  for (let i = 0; i < xArr.length; i++) {
    const xOut = xArr[i] < xBounds.lower || xArr[i] > xBounds.upper;
    const yOut = yArr[i] < yBounds.lower || yArr[i] > yBounds.upper;
    if (xOut || yOut) {
      nRemoved++;
    } else {
      xClean.push(xArr[i]);
      yClean.push(yArr[i]);
    }
  }

  return { xClean, yClean, nRemoved };
}

// ── Pearson's r ───────────────────────────────────────────────────────────────
function pearsonR(xArr, yArr) {
  const n = xArr.length;
  if (n < 2) return null;

  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
  for (let i = 0; i < n; i++) {
    sumX  += xArr[i];
    sumY  += yArr[i];
    sumXY += xArr[i] * yArr[i];
    sumX2 += xArr[i] * xArr[i];
    sumY2 += yArr[i] * yArr[i];
  }

  const num   = n * sumXY - sumX * sumY;
  const denom = Math.sqrt((n * sumX2 - sumX ** 2) * (n * sumY2 - sumY ** 2));
  return denom === 0 ? null : num / denom;
}

// ── Spearman's ρ ──────────────────────────────────────────────────────────────
function rankArray(arr) {
  const indexed = arr.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => a.v - b.v);

  const ranks = new Array(arr.length);
  let i = 0;
  while (i < indexed.length) {
    let j = i;
    while (j < indexed.length - 1 && indexed[j + 1].v === indexed[j].v) j++;
    const avgRank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) {
      ranks[indexed[k].i] = avgRank;
    }
    i = j + 1;
  }
  return ranks;
}

function spearmanRho(xArr, yArr) {
  if (xArr.length < 2) return null;
  const xRanks = rankArray(xArr);
  const yRanks = rankArray(yArr);
  return pearsonR(xRanks, yRanks);
}

// ── Interpret correlation strength ────────────────────────────────────────────
function interpretR(r) {
  if (r === null) return "Insufficient data";
  const abs = Math.abs(r);
  const dir = r >= 0 ? "positive" : "negative";
  if      (abs >= 0.7) return `Strong ${dir}`;
  else if (abs >= 0.4) return `Moderate ${dir}`;
  else if (abs >= 0.2) return `Weak ${dir}`;
  else                  return "Negligible";
}

// ── Build Correlations sheet ──────────────────────────────────────────────────
function buildCorrelationSheet() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();

  const conSheet = ss.getSheetByName("Consolidated");
  if (!conSheet) {
    SpreadsheetApp.getUi().alert('No "Consolidated" sheet found. Run "Consolidate by Date" first.');
    return;
  }

  const data    = conSheet.getDataRange().getValues();
  const headers = data[0];

  const colIdx = {};
  headers.forEach((h, i) => { colIdx[h] = i; });

  const vocalizationCols = [
    "Rods_Fighting_Calls_Per_Hour",
    "Rods_Talking_Calls_Per_Hour",
    "Straws_Fighting_Calls_Per_Hour",
    "Straws_Fighting_Talking_Calls_Per_Hour",
    "Straws_Talking_Calls_Per_Hour",
    "Straws_Want_Food_Calls_Per_Hour",
    "Rods_Total_Calls_Per_Hour",
    "Straws_Total_Calls_Per_Hour",
    "Combined_Total_Calls_Per_Hour"
  ];

  const targetCol = "Adjusted consumption";

  if (colIdx[targetCol] === undefined) {
    SpreadsheetApp.getUi().alert('"Adjusted consumption" column not found in Consolidated sheet.');
    return;
  }

  const results = [];

  vocalizationCols.forEach(vocCol => {
    if (colIdx[vocCol] === undefined) {
      results.push([vocCol, "Column not found", "", "", "", ""]);
      return;
    }

    const xRaw = [], yRaw = [];
    for (let r = 1; r < data.length; r++) {
      const x = parseFloat(data[r][colIdx[vocCol]]);
      const y = parseFloat(data[r][colIdx[targetCol]]);
      if (!isNaN(x) && !isNaN(y)) {
        xRaw.push(x);
        yRaw.push(y);
      }
    }

    const { xClean, yClean, nRemoved } = removeOutliers(xRaw, yRaw);

    const pr  = pearsonR(xClean, yClean);
    const rho = spearmanRho(xClean, yClean);
    const nPairs = xClean.length;

    results.push([
      vocCol,
      pr  !== null ? parseFloat(pr.toFixed(4))  : "N/A",
      rho !== null ? parseFloat(rho.toFixed(4)) : "N/A",
      nPairs,
      nRemoved,
      interpretR(pr),
      interpretR(rho)
    ]);
  });

  results.sort((a, b) => {
    const rA = typeof a[1] === "number" ? Math.abs(a[1]) : -1;
    const rB = typeof b[1] === "number" ? Math.abs(b[1]) : -1;
    return rB - rA;
  });

  let corrSheet = ss.getSheetByName("Correlations");
  if (corrSheet) {
    corrSheet.clearContents();
    corrSheet.clearFormats();
  } else {
    corrSheet = ss.insertSheet("Correlations");
  }

  corrSheet.getRange(1, 1).setValue("Pearson's r & Spearman's ρ — Vocalization vs. Adjusted Consumption (outliers removed via IQR)");
  corrSheet.getRange(1, 1).setFontWeight("bold").setFontSize(12);

  corrSheet.getRange(2, 1).setValue("Outliers removed where x or y fell outside Q1 − 1.5×IQR or Q3 + 1.5×IQR");
  corrSheet.getRange(2, 1).setFontStyle("italic").setFontColor("#666666");

  const colHeaders = ["Vocalization Metric", "Pearson's r", "Spearman's ρ", "n (pairs)", "Outliers Removed", "Pearson Strength", "Spearman Strength"];
  corrSheet.getRange(3, 1, 1, colHeaders.length).setValues([colHeaders]);
  corrSheet.getRange(3, 1, 1, colHeaders.length).setFontWeight("bold").setBackground("#d9e1f2");

  corrSheet.getRange(4, 1, results.length, colHeaders.length).setValues(results);

  const prRange  = corrSheet.getRange(4, 2, results.length, 1);
  const rhoRange = corrSheet.getRange(4, 3, results.length, 1);
  const rules    = [];

  [prRange, rhoRange].forEach(range => {
    rules.push(
      SpreadsheetApp.newConditionalFormatRule()
        .whenNumberGreaterThanOrEqualTo(0.7)
        .setBackground("#c6efce").setFontColor("#276221")
        .setRanges([range]).build()
    );
    rules.push(
      SpreadsheetApp.newConditionalFormatRule()
        .whenNumberBetween(0.4, 0.699)
        .setBackground("#ebf5eb").setFontColor("#276221")
        .setRanges([range]).build()
    );
    rules.push(
      SpreadsheetApp.newConditionalFormatRule()
        .whenNumberLessThanOrEqualTo(-0.7)
        .setBackground("#ffc7ce").setFontColor("#9c0006")
        .setRanges([range]).build()
    );
    rules.push(
      SpreadsheetApp.newConditionalFormatRule()
        .whenNumberBetween(-0.699, -0.4)
        .setBackground("#fce8e6").setFontColor("#9c0006")
        .setRanges([range]).build()
    );
  });

  corrSheet.setConditionalFormatRules(rules);
  corrSheet.autoResizeColumns(1, colHeaders.length);

  SpreadsheetApp.getUi().alert(
    "Done! Correlations written to the 'Correlations' sheet.\n" +
    results.length + " metrics correlated against Adjusted consumption.\n" +
    "Check the 'Outliers Removed' column to see how many data points were excluded per metric."
  );
}
// ── Linear regression (slope + intercept) ─────────────────────────────────────
function linearRegression(xArr, yArr) {
  const n = xArr.length;
  if (n < 2) return null;

  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  for (let i = 0; i < n; i++) {
    sumX  += xArr[i];
    sumY  += yArr[i];
    sumXY += xArr[i] * yArr[i];
    sumX2 += xArr[i] * xArr[i];
  }

  const denom = n * sumX2 - sumX * sumX;
  if (denom === 0) return null;

  const slope     = (n * sumXY - sumX * sumY) / denom;
  const intercept = (sumY - slope * sumX) / n;
  return { slope, intercept };
}
// ── Build Scatter Plot Sheets ─────────────────────────────────────────────────
/**
 * For each vocalization metric, creates a dedicated sheet with:
 *   - The cleaned (outlier-removed) x/y data in columns A & B
 *   - A scatter chart of that data embedded in the sheet
 *   - Pearson's r and Spearman's ρ displayed below the chart
 * All graph sheets are grouped with a "Graph_" prefix so they're easy to find.
 * Existing graph sheets are deleted and recreated fresh each run.
 */
function buildScatterPlots() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();

  const conSheet = ss.getSheetByName("Consolidated");
  if (!conSheet) {
    SpreadsheetApp.getUi().alert('No "Consolidated" sheet found. Run "Consolidate by Date" first.');
    return;
  }

  const data    = conSheet.getDataRange().getValues();
  const headers = data[0];
  const colIdx  = {};
  headers.forEach((h, i) => { colIdx[h] = i; });

  const vocalizationCols = [
    "Rods_Fighting_Calls_Per_Hour",
    "Rods_Talking_Calls_Per_Hour",
    "Straws_Fighting_Calls_Per_Hour",
    "Straws_Fighting_Talking_Calls_Per_Hour",
    "Straws_Talking_Calls_Per_Hour",
    "Straws_Want_Food_Calls_Per_Hour",
    "Rods_Total_Calls_Per_Hour",
    "Straws_Total_Calls_Per_Hour",
    "Combined_Total_Calls_Per_Hour"
  ];

  const targetCol = "Adjusted consumption";

  if (colIdx[targetCol] === undefined) {
    SpreadsheetApp.getUi().alert('"Adjusted consumption" column not found in Consolidated sheet.');
    return;
  }

  // ── Pre-compute stats for all cols and sort by absolute Pearson's r ───────
  const colsWithR = vocalizationCols
    .filter(vocCol => colIdx[vocCol] !== undefined)
    .map(vocCol => {
      const xRaw = [], yRaw = [];
      for (let r = 1; r < data.length; r++) {
        const x = parseFloat(data[r][colIdx[vocCol]]);
        const y = parseFloat(data[r][colIdx[targetCol]]);
        if (!isNaN(x) && !isNaN(y)) { xRaw.push(x); yRaw.push(y); }
      }
      const { xClean, yClean, nRemoved } = removeOutliers(xRaw, yRaw);
      const pr  = pearsonR(xClean, yClean);
      const rho = spearmanRho(xClean, yClean);
      return { vocCol, xClean, yClean, nRemoved, pr, rho };
    })
    .sort((a, b) => Math.abs(b.pr ?? 0) - Math.abs(a.pr ?? 0));

  // ── Delete any old Graph_ sheets ──────────────────────────────────────────
  const toDelete = ss.getSheets().filter(s => s.getName().startsWith("Graph_"));
  toDelete.forEach(s => ss.deleteSheet(s));

  // ── Create one sheet + chart per vocalization column ──────────────────────
  colsWithR.forEach(({ vocCol, xClean, yClean, nRemoved, pr, rho }) => {

    const shortName = vocCol.replace(/_Calls_Per_Hour$/, "").replace(/_Per_Hour$/, "");
    const sheetName = ("Graph_" + shortName).substring(0, 100);

    const gs = ss.insertSheet(sheetName);

    gs.getRange(1, 1).setValue(vocCol);
    gs.getRange(1, 2).setValue(targetCol);
    gs.getRange(1, 1, 1, 2).setFontWeight("bold").setBackground("#d9e1f2");

    if (xClean.length > 0) {
      const dataRows = xClean.map((x, i) => [x, yClean[i]]);
      gs.getRange(2, 1, dataRows.length, 2).setValues(dataRows);
    }

    const statsRow = xClean.length + 3;
    const reg = linearRegression(xClean, yClean);
    const eqStr = reg
      ? `y = ${reg.slope.toFixed(4)}x ${reg.intercept >= 0 ? "+ " + reg.intercept.toFixed(4) : "− " + Math.abs(reg.intercept).toFixed(4)}`
      : "N/A";

    gs.getRange(statsRow,     1).setValue("Trendline Equation:");
    gs.getRange(statsRow,     2).setValue(eqStr);
    gs.getRange(statsRow + 1, 1).setValue("Pearson's r:");
    gs.getRange(statsRow + 1, 2).setValue(pr  !== null ? parseFloat(pr.toFixed(4))  : "N/A");
    gs.getRange(statsRow + 2, 1).setValue("Spearman's ρ:");
    gs.getRange(statsRow + 2, 2).setValue(rho !== null ? parseFloat(rho.toFixed(4)) : "N/A");
    gs.getRange(statsRow + 3, 1).setValue("Pearson Strength:");
    gs.getRange(statsRow + 3, 2).setValue(interpretR(pr));
    gs.getRange(statsRow + 4, 1).setValue("Spearman Strength:");
    gs.getRange(statsRow + 4, 2).setValue(interpretR(rho));
    gs.getRange(statsRow + 5, 1).setValue("n (after outlier removal):");
    gs.getRange(statsRow + 5, 2).setValue(xClean.length);
    gs.getRange(statsRow + 6, 1).setValue("Outliers removed:");
    gs.getRange(statsRow + 6, 2).setValue(nRemoved);

    gs.getRange(statsRow, 1, 7, 1).setFontWeight("bold");
    gs.autoResizeColumns(1, 2);

    if (xClean.length >= 2) {
      const dataRange = gs.getRange(1, 1, xClean.length + 1, 2);
      const prLabel  = pr  !== null ? pr.toFixed(4)  : "N/A";
      const rhoLabel = rho !== null ? rho.toFixed(4) : "N/A";

      const chart = gs.newChart()
        .setChartType(Charts.ChartType.SCATTER)
        .addRange(dataRange)
        .setOption("title", `${shortName.replace(/_/g, " ")} vs. Adjusted Consumption`)
        .setOption("titleTextStyle", { fontSize: 13, bold: true })
        .setOption("hAxis", {
          title: vocCol.replace(/_/g, " "),
          titleTextStyle: { italic: false }
        })
        .setOption("vAxis", {
          title: "Adjusted Consumption",
          titleTextStyle: { italic: false }
        })
        .setOption("legend", { position: "none" })
        .setOption("pointSize", 6)
        .setOption("colors", ["#4472c4"])
        .setOption("trendlines", {
          0: {
            type: "linear",
            color: "#e06c75",
            lineWidth: 2,
            opacity: 0.8,
            showR2: true,
            visibleInLegend: true,
            labelInLegend: `Trendline (r=${prLabel}, ρ=${rhoLabel})`
          }
        })
        .setPosition(1, 4, 0, 0)
        .setOption("width",  500)
        .setOption("height", 380)
        .build();

      gs.insertChart(chart);
    } else {
      gs.getRange(1, 4).setValue("Not enough data to plot a chart (need ≥ 2 points after outlier removal).");
    }
  });

  SpreadsheetApp.getUi().alert(
    "Done! " + colsWithR.length + " scatter plot sheets created, ordered strongest → weakest correlation.\n" +
    "Look for sheets starting with 'Graph_' in your spreadsheet tabs."
  );
}

// ── Menu ──────────────────────────────────────────────────────────────────────
function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu("🦇 Bat Data")
    .addItem("Consolidate by Date",   "consolidateByDate")
    .addItem("Build Correlation Sheet", "buildCorrelationSheet")
    .addItem("Build Scatter Plots",   "buildScatterPlots")
    .addToUi();
}
