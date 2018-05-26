cat streetview.html | perl -e 'while ($l=<>) { $text .= $l};  @links = $text =~ /"gallery-item-link" href="([^"]+)"/gsm; use Data::Dumper; print $_ . "\n" for @links' > streetview.txt
perl parse.pl > results.tsv
cut -f1 results.tsv | grep -v 'panoid' | while read i; do echo "https://geo0.ggpht.com/cbk?output=thumbnail&thumb=2&panoid=$i" ; done > urls.txt
