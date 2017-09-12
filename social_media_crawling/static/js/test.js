var forma = function(show){
		var a = document.getElementById(show);
		a.style.display = (a.style.display == 'block') ? 'none' : 'block';
	}
var formb = function(show, hide){
		var a = document.getElementById('ApiOpt')
		a.style.display = (a.style.display == 'none')?'block' : 'none';
		var b = document.getElementById('SrcOpt')
		b.style.display = (a.style.display == 'block')?'none' : 'block';	
	}
	
var c = [];

var bsd = function(){		
	for (i=0; i<c.length;i++){
	c.pop()
	}
	var a = document.getElementById('country_dropdown').value;
	c.push(a);
}	
var asd = function(){
	var a = document.getElementById('country_dropdown').value;
	var b = a.split(" ").length;
	var d = c[0].split(" ").length;	
	if (b==1&d==1){
	forma(c)
	forma(a)
	}
	else if(b==1&d!=1){
	var words = c[0].split(" ");
	forma(words[0])
	forma(a)
	}
	else if(b!=1&d!=1){
	var words = c[0].split(" ");
	forma(c)
	var words = a.split(" ");
	forma(a)
	}
	else{
	var words = a.split(" ");
	forma(words[0])
	forma(c)
	}

}

var dateboundary = function(){
	
}