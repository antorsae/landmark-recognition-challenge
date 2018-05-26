#!/usr/bin/perl
use strict;
use Data::Dumper;
use JSON qw(decode_json);
use WWW::Mechanize();

my $mech = WWW::Mechanize->new();
my @keys = qw(panoid id title lat lng pitch heading);

print join("\t", @keys) . "\n";
open(my $fh, "<", "streetview.txt");
binmode($fh, ":utf8");

while (my $line = <$fh>) {
	$line =~ s/\n//;
	$line =~ s/\#//;
	$line =~ s/\/.*//;
	$mech->get("https://www.google.com/streetview/feed/gallery/collection/" . $line . ".json");
	my $json = decode_json($mech->content());
	for my $key (keys %$json) {
		$json->{$key}->{id} = $key;
		my @list = map { $json->{$key}->{$_} } @keys;
		print join("\t", @list) . "\n";
	}
}
close($fh);
