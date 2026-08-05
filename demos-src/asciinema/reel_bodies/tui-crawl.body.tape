Hide
Type "export LILBEE_DATA=/root/data-crawl && source /root/kit/env.sh && rm -rf /root/data-crawl && lilbee"
Enter
Sleep __BOOT__s
Show
Sleep 1500ms
# /crawl with no arg opens the modal; URL input auto-focuses.
Type "/crawl"
Sleep 400ms
Enter
Sleep 1500ms
Type "https://en.wikipedia.org/wiki/Chevrolet_Caprice"
Sleep 1500ms
# Tab: URL -> Recursive -> Advanced collapsible; expand, set depth 0.
Tab
Sleep 800ms
Tab
Sleep 600ms
Enter
Sleep 1s
Tab
Sleep 400ms
Type "0"
Sleep 800ms
Tab
Sleep 800ms
Tab
Sleep 500ms
Enter
Sleep 4s
# Drop to NORMAL so `t` fires the Tasks binding; watch the crawl land.
Escape
Sleep 800ms
Type "t"
Sleep 3s
Hide
Sleep __CRAWLREM__s
Show
Sleep 2500ms
# Tasks -> Chat; re-enter INSERT, ask the single-fact question.
Type "]"
Sleep 1000ms
Type "i"
Sleep 400ms
Type "When was the 9C1 Caprice police package introduced?"
Sleep 600ms
Enter
# Show the stream start, hide the bulk, return for the settled cited answer.
Sleep 4s
Hide
Sleep __GENREM__s
Show
Sleep 1200ms
Escape
Sleep 400ms
Type "G"
Sleep 2500ms
Screenshot __REEL__.png