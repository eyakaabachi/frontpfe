
var ageGroups = ["18-24", "25-29", "30-34", "35-39","40-44","45-49","50+"];
var ageGroupCounts = [24192, 19746, 22917, 24545, 25942, 30276, 293832];
var incomeGroups = ["less than $15,000", "15,000 - $25,000", "$25,000 - $35,000", "$35,000 - $50,000","$50,000 or more"];
var incomeGroupCounts = [38048, 59174, 39235, 52052, 252947];
// Bar Chart Example
var ctx = document.getElementById("myBarChart");
const myBarChart = new Chart(ctx, {
  type: 'bar',
  data: {
    labels: ageGroups, // Use the age group labels as chart labels
    datasets: [{
      label: "Number of Patients",
      backgroundColor: "#4e73df",
      hoverBackgroundColor: "#2e59d9",
      borderColor: "#4e73df",
      data: ageGroupCounts, // Use the age group counts as chart data
    }],
  },
  options: {
    maintainAspectRatio: false,
    layout: {
      padding: {
        left: 10,
        right: 25,
        top: 25,
        bottom: 0
      }
    },
    legend: {
      display: true
    },
    tooltips: {
      titleMarginBottom: 10,
      titleFontColor: '#6e707e',
      titleFontSize: 14,
      backgroundColor: "rgb(255,255,255)",
      bodyFontColor: "#858796",
      borderColor: '#dddfeb',
      borderWidth: 1,
      xPadding: 15,
      yPadding: 15,
      displayColors: false,
      caretPadding: 10,
      callbacks: {
        label: function(tooltipItem, chart) {
          var datasetLabel = chart.datasets[tooltipItem.datasetIndex].label || '';
          return datasetLabel + ': ' + number_format(tooltipItem.yLabel);
        }
      }
    },
  }
});

var ctx = document.getElementById("myBarChart2");
const myBarChart2 = new Chart(ctx, {
  type: 'bar',
  data: {
    labels: incomeGroups, // Use the age group labels as chart labels
    datasets: [{
      label: "Number of Patients",
      backgroundColor: "#36b9cc",
      hoverBackgroundColor: "#2c9faf",
      borderColor: "#4e73df",
      data: incomeGroupCounts, // Use the age group counts as chart data
    }],
  },
  options: {
    maintainAspectRatio: false,
    layout: {
      padding: {
        left: 10,
        right: 25,
        top: 25,
        bottom: 0
      }
    },
    legend: {
      display: true
    },
    tooltips: {
      titleMarginBottom: 10,
      titleFontColor: '#6e707e',
      titleFontSize: 14,
      backgroundColor: "rgb(255,255,255)",
      bodyFontColor: "#858796",
      borderColor: '#dddfeb',
      borderWidth: 1,
      xPadding: 15,
      yPadding: 15,
      displayColors: false,
      caretPadding: 10,
      callbacks: {
        label: function(tooltipItem, chart) {
          var datasetLabel = chart.datasets[tooltipItem.datasetIndex].label || '';
          return datasetLabel + ': ' + number_format(tooltipItem.yLabel);
        }
      }
    },
  }
});
