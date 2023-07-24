// Call the dataTables jQuery plugin

var ctx = document.getElementById("dataTable");
let table = new DataTable('#dataTable', {
    paging: true,
    searching: true,
    ordering:  true
});
