#!/bin/bash
pdf_dir="`pwd`/pdfs/" 
file="book_urls.txt"

while read url; do 
    google-chrome -incognito &
    sleep 5 
    xdotool type "$url"
    xdotool key KP_Enter
    sleep 10
    xdotool key Home
    xdotool type "$pdf_dir" 
    xdotool key KP_Enter
    sleep 5 
    xdotool key Ctrl+w
done < $file
