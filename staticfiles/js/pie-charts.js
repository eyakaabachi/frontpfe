// Your llcp2015 dataset sex data (replace these with your actual data)
var maleCount = 186938;
var femaleCount = 254518;

// Pie Chart Example
var ctx = document.getElementById("myPieChart");
var myPieChart = new Chart(ctx, {
  type: 'doughnut',
  data: {
    labels: ["Male", "Female"],
    datasets: [{
      data: [maleCount, femaleCount],
      backgroundColor: ['#4e73df', '#1cc88a'],
      hoverBackgroundColor: ['#2e59d9', '#17a673'],
      hoverBorderColor: "rgba(234, 236, 244, 1)",
    }],
  },
  options: {
    maintainAspectRatio: false,
    tooltips: {
      backgroundColor: "rgb(255,255,255)",
      bodyFontColor: "#858796",
      borderColor: '#dddfeb',
      borderWidth: 1,
      xPadding: 15,
      yPadding: 15,
      displayColors: false,
      caretPadding: 10,
    },
    legend: {
      display: true, // Set to true if you want to display the legend
      position: 'bottom', // Position of the legend ('top', 'left', 'bottom', 'right')
    },
    cutoutPercentage: 80,
  },
});



var marriedCount = 236306;
var singleCount = 80295;
var divorcedCount = 59406;
var widowedCount = 56481;
var separatedCount = 8968;
var ctx = document.getElementById("myPieChart2");
var myPieChart2 = new Chart(ctx, {
  type: 'doughnut',
  data: {
    labels: ["Married", "Single", "Divorced", "Widowed","Separated"],
    datasets: [{
      data: [marriedCount, singleCount, divorcedCount, widowedCount, separatedCount],
      backgroundColor: ['#36b9cc', '#1cc88a', '#f6c23e', '#4e73df', '#858796'],
      hoverBackgroundColor: ['#2c9faf', '#17a673', '#f5b316', '#2e59d9'],
      hoverBorderColor: "rgba(234, 236, 244, 1)",
    }],
  },
  options: {
    maintainAspectRatio: false,
    tooltips: {
      backgroundColor: "rgb(255,255,255)",
      bodyFontColor: "#858796",
      borderColor: '#dddfeb',
      borderWidth: 1,
      xPadding: 15,
      yPadding: 15,
      displayColors: false,
      caretPadding: 10,
    },
    legend: {
      display: true, // Set to true if you want to display the legend
      position: 'bottom', // Position of the legend ('top', 'left', 'bottom', 'right')
    },
    cutoutPercentage: 80,
  },
});


var eduGroupCounts = [609, 11187, 145690, 283970];
var ctx = document.getElementById("myPieChart3");
var myPieChart3 = new Chart(ctx, {
  type: 'doughnut',
  data: {
    labels: ["Uneducated", "Elementary", "High School", "College"],
    datasets: [{
      data: eduGroupCounts,
      backgroundColor: ['#36b9cc', '#1cc88a', '#f6c23e', '#4e73df', '#858796'],
      hoverBackgroundColor: ['#2c9faf', '#17a673', '#f5b316', '#2e59d9'],
      hoverBorderColor: "rgba(234, 236, 244, 1)",
    }],
  },
  options: {
    maintainAspectRatio: false,
    tooltips: {
      backgroundColor: "rgb(255,255,255)",
      bodyFontColor: "#858796",
      borderColor: '#dddfeb',
      borderWidth: 1,
      xPadding: 15,
      yPadding: 15,
      displayColors: false,
      caretPadding: 10,
    },
    legend: {
      display: true, // Set to true if you want to display the legend
      position: 'bottom', // Position of the legend ('top', 'left', 'bottom', 'right')
    },
    cutoutPercentage: 80,
  },
});