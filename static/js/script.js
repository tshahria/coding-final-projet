$(function(){
	$('button').click(function(){
		$.ajax({
			url: '/signUpUser',
			data: $('form').serialize(),
			type: 'POST',
			success: function(response){
				let res = JSON.parse(response);
				$("#marker").text(res.status);
			},
			error: function(error){
				console.log(error);
			}
		});
	});
});
