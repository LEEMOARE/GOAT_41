<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
    <%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<!DOCTYPE HTML>
<!--
	Dimension by HTML5 UP
	html5up.net | @ajlkn
	Free for personal and commercial use under the CCA 3.0 license (html5up.net/license)
-->
<html>
	<head>
		<link rel="icon" href="https://s3.ap-northeast-2.amazonaws.com/perfume.projact/lungs.ico">
		<title>GOAT.SNPPY</title>
		<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no" />
		<link href="https://s3.ap-northeast-2.amazonaws.com/perfume.projact/pro/new.css" rel="stylesheet" type="text/css">
		<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
		<noscript><link href="https://s3.ap-northeast-2.amazonaws.com/perfume.projact/pro/new.css" rel="stylesheet" type="text/css"></noscript>
	</head>
	<body class="is-preload">

		<!-- Wrapper -->
			<div id="wrapper">
				

				<!-- Header -->
					<header id="header">
					
						<div class="logo">
							<span class="fas fa-lungs" style="font-size: 30px;"></span>
						</div>
						<div class="content">
							<div class="inner">
								<h1>GOAT</h1>
								<p>Portable X-ray for CNN-based Lesion Localization</p>
							</div>
						</div>
						<nav>
							<ul>
								<li><a href="#intro">Intro</a></li>
								<li><a href="#upload">Upload</a></li>
								<li><a href="#login">Login</a></li>
								<li><a href="#register">Register</a></li>
								<!--<li><a href="#elements">Elements</a></li>-->
							</ul>
						</nav>
					</header>

				<!-- Main -->
					<div id="main">

						<!-- Intro -->
							<article id="intro">
								<h2 class="major">Intro</h2>
								<span class="image main"><img src="https://s3.ap-northeast-2.amazonaws.com/perfume.projact/pro/pic1.png">
</span>
								<p>제품설명하기!</p>
							</article>

						<!-- Upload -->
							<article id="upload">
								<h2 class="major">Upload</h2>
								<span class="image main"><img src='https://s3.ap-northeast-2.amazonaws.com/perfume.projact/pro/pic2.png'></span>
								<form action="/imageupload.do" method="post" enctype="multipart/form-data">
    								<input type="file" name="image" />
									<input type="submit" value="Upload" />
								</form>
								<p>사진업로드 하는데 로그인이 되어 있어야 할 수 있도록</p>
							</article>

						<!-- Login -->
							<article id="login">
								<h2 class="major">Login</h2>
								<form method="post" action="#">
									<div class="fields">
										<div class="field half">
											<label for="userid">User-ID</label>
											<div class="input-container">
												<input type="text" name="userid" id="userid" placeholder="Enter your User ID" />
											</div>
										</div>
										<div class="field half">
											<label for="password">PASSWORD</label>
											<div class="input-container">
												<input type="password" name="password" id="password" placeholder="Enter your Password" />
											</div>
										</div>
									</div>
									<ul class="actions">
										<li><input type="submit" value="로그인" class="primary" /></li>
										<li><input type="reset" value="초기화" /></li>
									</ul>
								</form>
							</article>

						<!-- Register -->
							<article id="register">
								<h2 class="major">Register</h2>
								<form method="post" action="#" id="register-form">
									<div class="fields">
										<div class="field">
											<label for="user-id">User ID</label>
											<input type="text" name="user-id" id="user-id" placeholder="Enter User ID" />
										</div>
										<div class="field">
											<label for="password">Password</label>
											<input type="password" name "password" id="password" placeholder="Enter Password" />
										</div>
										<div class="field">
											<label for="confirm-password">Confirm Password</label>
											<input type="password" name="confirm-password" id="confirm-password" placeholder="Confirm Password" />
										</div>
										<div class="field">
											<label for="medical-number">Medical Number (4 digits)</label>
											<input type="text" name="medical-number" id="medical-number" placeholder="Enter 4-digit Medical Number" />
										</div>
										<div class="field">
											<label for="full-name">Full Name</label>
											<input type="text" name="full-name" id="full-name" placeholder="Enter Full Name" />
										</div>
									</div>
									<ul class="actions">
										<li><input type="submit" value="Register" class="primary" /></li>
										<li><input type="reset" value="Reset" /></li>
									</ul>
								</form>
								<ul class="icons">
									<li><a href="#" class="icon brands fa-twitter"><span class="label">Twitter</span></a></li>
									<li><a href="#" class="icon brands fa-facebook-f"><span class="label">Facebook</span></a></li>
									<li><a href="#" class="icon brands fa-instagram"><span class="label">Instagram</span></a></li>
									<li><a href="#" class="icon brands fa-github"><span class="label">GitHub</span></a></li>
								</ul>
							</article>

							<script>
								// 의료번호 입력 필드
								const medicalNumberField = document.getElementById("medical-number");
							
								// 입력 값이 숫자인지 확인하는 함수
								function isNumeric(value) {
									return /^\d+$/.test(value);
								}
							
								// 의료번호 필드의 입력 이벤트 처리
								medicalNumberField.addEventListener("input", function () {
									const inputValue = medicalNumberField.value;
									if (!isNumeric(inputValue)) {
										alert("의료번호는 숫자로만 입력해야 합니다.");
										medicalNumberField.value = "";
									}
								});

								function imageupload() {
								    var form = document.getElementById('register-form');
								    var formData = new FormData(form);
								    var xhr = new XMLHttpRequest();
								    xhr.open('POST', 'upload', true);
								    xhr.onload = function () {
								        if (xhr.status === 200 || xhr.status === 201) {
								            console.log(xhr.responseText);
								        } else {
								            console.error(xhr.responseText);
								        }
								    };
								    xhr.send(formData);
								    
							</script>

						<!-- Elements -->
							<article id="elements">
								<h2 class="major">Elements</h2>

								<section>
									<h3 class="major">Text</h3>
									<p>This is <b>bold</b> and this is <strong>strong</strong>. This is <i>italic</i> and this is <em>emphasized</em>.
									This is <sup>superscript</sup> text and this is <sub>subscript</sub> text.
									This is <u>underlined</u> and this is code: <code>for (;;) { ... }</code>. Finally, <a href="#">this is a link</a>.</p>
									<hr />
									<h2>Heading Level 2</h2>
									<h3>Heading Level 3</h3>
									<h4>Heading Level 4</h4>
									<h5>Heading Level 5</h5>
									<h6>Heading Level 6</h6>
									<hr />
									<h4>Blockquote</h4>
									<blockquote>Fringilla nisl. Donec accumsan interdum nisi, quis tincidunt felis sagittis eget tempus euismod. Vestibulum ante ipsum primis in faucibus vestibulum. Blandit adipiscing eu felis iaculis volutpat ac adipiscing accumsan faucibus. Vestibulum ante ipsum primis in faucibus lorem ipsum dolor sit amet nullam adipiscing eu felis.</blockquote>
									<h4>Preformatted</h4>
									<pre><code>i = 0;

while (!deck.isInOrder()) {
    print 'Iteration ' + i;
    deck.shuffle();
    i++;
}

print 'It took ' + i + ' iterations to sort the deck.';</code></pre>
								</section>

								<section>
									<h3 class="major">Lists</h3>

									<h4>Unordered</h4>
									<ul>
										<li>Dolor pulvinar etiam.</li>
										<li>Sagittis adipiscing.</li>
										<li>Felis enim feugiat.</li>
									</ul>

									<h4>Alternate</h4>
									<ul class="alt">
										<li>Dolor pulvinar etiam.</li>
										<li>Sagittis adipiscing.</li>
										<li>Felis enim feugiat.</li>
									</ul>

									<h4>Ordered</h4>
									<ol>
										<li>Dolor pulvinar etiam.</li>
										<li>Etiam vel felis viverra.</li>
										<li>Felis enim feugiat.</li>
										<li>Dolor pulvinar etiam.</li>
										<li>Etiam vel felis lorem.</li>
										<li>Felis enim et feugiat.</li>
									</ol>
									<h4>Icons</h4>
									<ul class="icons">
										<li><a href="#" class="icon brands fa-twitter"><span class="label">Twitter</span></a></li>
										<li><a href="#" class="icon brands fa-facebook-f"><span class="label">Facebook</span></a></li>
										<li><a href="#" class="icon brands fa-instagram"><span class="label">Instagram</span></a></li>
										<li><a href="#" class="icon brands fa-github"><span class="label">Github</span></a></li>
									</ul>

									<h4>Actions</h4>
									<ul class="actions">
										<li><a href="#" class="button primary">Default</a></li>
										<li><a href="#" class="button">Default</a></li>
									</ul>
									<ul class="actions stacked">
										<li><a href="#" class="button primary">Default</a></li>
										<li><a href="#" class="button">Default</a></li>
									</ul>
								</section>

								<section>
									<h3 class="major">Table</h3>
									<h4>Default</h4>
									<div class="table">
										<table>
											<thead>
												<tr>
													<th>Name</th>
													<th>Description</th>
													<th>Price</th>
												</tr>
											</thead>
											<tbody>
												<tr>
													<td>Item One</td>
													<td>Ante turpis integer aliquet porttitor.</td>
													<td>29.99</td>
												</tr>
												<tr>
													<td>Item Two</td>
													<td>Vis ac commodo adipiscing arcu aliquet.</td>
													<td>19.99</td>
												</tr>
												<tr>
													<td>Item Three</td>
													<td> Morbi faucibus arcu accumsan lorem.</td>
													<td>29.99</td>
												</tr>
												<tr>
													<td>Item Four</td>
													<td>Vitae integer tempus condimentum.</td>
													<td>19.99</td>
												</tr>
												<tr>
													<td>Item Five</td>
													<td>Ante turpis integer aliquet porttitor.</td>
													<td>29.99</td>
												</tr>
											</tbody>
											<tfoot>
												<tr>
													<td colspan="2"></td>
													<td>100.00</td>
												</tr>
											</tfoot>
										</table>
									</div>

									<h4>Alternate</h4>
									<div class="table-wrapper">
										<table class="alt">
											<thead>
												<tr>
													<th>Name</th>
													<th>Description</th>
													<th>Price</th>
												</tr>
											</thead>
											<tbody>
												<tr>
													<td>Item One</td>
													<td>Ante turpis integer aliquet porttitor.</td>
													<td>29.99</td>
												</tr>
												<tr>
													<td>Item Two</td>
													<td>Vis ac commodo adipiscing arcu aliquet.</td>
													<td>19.99</td>
												</tr>
												<tr>
													<td>Item Three</td>
													<td> Morbi faucibus arcu accumsan lorem.</td>
													<td>29.99</td>
												</tr>
												<tr>
													<td>Item Four</td>
													<td>Vitae integer tempus condimentum.</td>
													<td>19.99</td>
												</tr>
												<tr>
													<td>Item Five</td>
													<td>Ante turpis integer aliquet porttitor.</td>
													<td>29.99</td>
												</tr>
											</tbody>
											<tfoot>
												<tr>
													<td colspan="2"></td>
													<td>100.00</td>
												</tr>
											</tfoot>
										</table>
									</div>
								</section>

								<section>
									<h3 class="major">Buttons</h3>
									<ul class="actions">
										<li><a href="#" class="button primary">Primary</a></li>
										<li><a href="#" class="button">Default</a></li>
									</ul>
									<ul class="actions">
										<li><a href="#" class="button">Default</a></li>
										<li><a href="#" class="button small">Small</a></li>
									</ul>
									<ul class="actions">
										<li><a href="#" class="button primary icon solid fa-download">Icon</a></li>
										<li><a href="#" class="button icon solid fa-download">Icon</a></li>
									</ul>
									<ul class="actions">
										<li><span class="button primary disabled">Disabled</span></li>
										<li><span class="button disabled">Disabled</span></li>
									</ul>
								</section>

								<section>
									<h3 class="major">Form</h3>
									<form method="post" action="#">
										<div class="fields">
											<div class="field half">
												<label for="demo-name">Name</label>
												<input type="text" name="demo-name" id="demo-name" value="" placeholder="Jane Doe" />
											</div>
											<div class="field half">
												<label for="demo-email">Email</label>
												<input type="email" name="demo-email" id="demo-email" value="" placeholder="jane@untitled.tld" />
											</div>
											<div class="field">
												<label for="demo-category">Category</label>
												<select name="demo-category" id="demo-category">
													<option value="">-</option>
													<option value="1">Manufacturing</option>
													<option value="1">Shipping</option>
													<option value="1">Administration</option>
													<option value="1">Human Resources</option>
												</select>
											</div>
											<div class="field half">
												<input type="radio" id="demo-priority-low" name="demo-priority" checked>
												<label for="demo-priority-low">Low</label>
											</div>
											<div class="field half">
												<input type="radio" id="demo-priority-high" name="demo-priority">
												<label for="demo-priority-high">High</label>
											</div>
											<div class="field half">
												<input type="checkbox" id="demo-copy" name="demo-copy">
												<label for="demo-copy">Email me a copy</label>
											</div>
											<div class="field half">
												<input type="checkbox" id="demo-human" name="demo-human" checked>
												<label for="demo-human">Not a robot</label>
											</div>
											<div class="field">
												<label for="demo-message">Message</label>
												<textarea name="demo-message" id="demo-message" placeholder="Enter your message" rows="6"></textarea>
											</div>
										</div>
										<ul class="actions">
											<li><input type="submit" value="Send Message" class="primary" /></li>
											<li><input type="reset" value="Reset" /></li>
										</ul>
									</form>
								</section>

							</article>

					</div>

				<!-- Footer -->
					<footer id="footer">
						<p class="copyright">&copy; Together: <a href='https://html5up.net' />">GOAT</a>.</p>
					</footer>

			</div>

		<!-- BG -->
			<div id="bg"></div>

		<!-- Scripts -->
			<script src="https://s3.ap-northeast-2.amazonaws.com/perfume.projact/pro/js/jquery.min.js"></script>
			<script src="https://s3.ap-northeast-2.amazonaws.com/perfume.projact/pro/js/browser.min.js"></script>
			<script src="https://s3.ap-northeast-2.amazonaws.com/perfume.projact/pro/js/breakpoints.min.js"></script>
			<script src="https://s3.ap-northeast-2.amazonaws.com/perfume.projact/pro/js/util.js"></script>
			<script src="https://s3.ap-northeast-2.amazonaws.com/perfume.projact/pro/js/main.js"></script>


	</body>
</html>
