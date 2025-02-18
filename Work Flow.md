1. First we need to integrate the data from the demos to a data frame 
2. API calls would help a lot, but looking into HLTV unofficial Node.js API we can't pull the same amount of data as we do using the downloaded demos
	1. https://github.com/gigobyte/HLTV
	2. We can actually pull **FullPlayerStats**, which is great
	3. The library is no longer actively maintained (High Risk)
3. Still working on getting all the raw data in one data frame, starting by Players
	1. https://docs.google.com/spreadsheets/d/1U0tUdHcdQmB2kUKX0ky-nCnBWXAoHpqDRs7D7HKdBIU/edit?gid=0#gid=0
	2. The initial idea is to replicate most of the **Players** tab in a data frame
	3. Rounds are an issue due to lack of relationship with the player, my first work around it was to assume that the player won't leave the team in the middle of the tournament
	4. Some tricky raw data
		1. Total Rounds
		2. CT Rounds Lost
		3. T Rounds Lost
		4. CT Rounds 
		5. T Rounds
		6. Die after getting kill
		7. Saves
		8. CT Saves
		9. T Saves
		10. First Kills Raw
		11. Grenades thrown 

kast = percentage 
adr = average damage per round
