<?php
// Allow access from anywhere. Can be domains or * (any)
header('Access-Control-Allow-Origin: *');

// Allow these methods of data retrieval
header('Access-Control-Allow-Methods: POST, GET, OPTIONS');

// Allow these header types
header('Access-Control-Allow-Headers: Origin, X-Requested-With, Content-Type, Accept');

// Create our data object in that crazy unique PHP way
$arr = array(array("artist" => 1, "artistName" => "AC/DC", "genre" => "Rock", "image" => "acdc.jpg", "description" => "AC/DC are an Australian hard rock band, formed in 1973 by brothers Malcolm and Angus Young. To date they are one of the highest-grossing bands of all time."));

// Return as JSON
echo json_encode($arr);
?>
