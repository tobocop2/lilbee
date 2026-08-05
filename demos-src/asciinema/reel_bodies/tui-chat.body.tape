Hide
Type "export LILBEE_DATA=/root/data-chat && source /root/kit/env.sh && rm -rf /root/data-chat && lilbee"
Enter
Sleep __BOOT__s
Show
Sleep 1500ms
# add the car's service manual to the knowledge base on camera
Type "/add /root/cv-manual.pdf"
Sleep 700ms
Enter
Sleep 6s
# finish indexing the PDF off camera
Hide
Sleep __INDEXREM__s
Show
Sleep 1200ms
# glance at the task center so the indexed doc is visible, then back
Escape
Sleep 400ms
Type "t"
Sleep 3s
Escape
Sleep 700ms
# open the fleet drawer and leave it up for the answer
Ctrl+g
Sleep 3s
# ask one grounded question (Escape+i = deterministic insert entry, no stray 'i')
Escape
Sleep 400ms
Type "i"
Sleep 400ms
Type@15ms "My Crown Vic has a dead battery. Walk me through jump-starting it safely: the exact cable connection order and any warnings specific to this car."
Sleep 900ms
Enter
Sleep __GEN__s
# scroll to the bottom so the full settled answer is on screen
Escape
Sleep 500ms
Type "G"
Sleep 2500ms
# answer is done: close the drawer
Ctrl+g
Sleep 1500ms
Screenshot __REEL__.png
