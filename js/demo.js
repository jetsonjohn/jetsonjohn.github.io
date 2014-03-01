function create_zip() {
  var fileURL = jQuery('#file').attr('href');
  var request = jQuery.ajax({
    url: fileURL,
    type: "GET",
    contentType: "application/pdf",
    mimeType:'text/plain; charset=x-user-defined' // <-[1]
  });     

  request.done(function( data ) {
    var zip = new JSZip();
    zip.file("my_file.pdf", data, { binary: true }); // <- [2]
    content = zip.generate();
    location.href = "data:application/zip;base64," + content;
  });       
}
