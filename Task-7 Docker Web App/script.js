function docker(){
    // alert("hello")

var i  = document.getElementById("command").value
// print(i);
var hr = new XMLHttpRequest();
// hr.onload
hr.open("Get","http://192.168.1.7/cgi-bin/app.py?cmd=" + i ,true);
hr.send();
hr.onload = function(){
    var output = hr.responseText;
    document.getElementById('d1').innerHTML = output;
}

}

