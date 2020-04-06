from game_model import *
from namelist import namelist

def interpret_state(state, our_active, opponent_active, our_team, opponent_team):
    assert(len(state) == N)
    our_side = ""
    opponent_side = ""
    neutral_side = ""
    our_side += our_team[our_active]
    opponent_side += opponent_team[opponent_active]

    for i in range(TEAM_SIZE):
        our_side += our_team[i] + str(state[i+OFFSET_HEALTH])
        for j in range(NUM_STATUS_CONDITIONS):
            if (state[OFFSET_STATUS_CONDITIONS + i*NUM_STATUS_CONDITIONS + j] != 0):
                our_side += STATUS_LOOKUP[j] + str(state[OFFSET_STATUS_CONDITIONS + i*NUM_STATUS_CONDITIONS + j])

        opponent_side += opponent_team[i] + str(state[i+TEAM_SIZE+OFFSET_HEALTH])
        for j in range(NUM_STATUS_CONDITIONS):
            if (state[OFFSET_STATUS_CONDITIONS+ NUM_STATUS_CONDITIONS * TEAM_SIZE + i*NUM_STATUS_CONDITIONS + j] != 0):
                opponent_side += STATUS_LOOKUP[j] + str(state[OFFSET_STATUS_CONDITIONS+ NUM_STATUS_CONDITIONS * TEAM_SIZE + i*NUM_STATUS_CONDITIONS + j])

    our_side += "boosts:"
    opponent_side += "boosts:"
    for i in range(NUM_STAT_BOOSTS):
        our_side += str(state[OFFSET_STAT_BOOSTS + i]) + "|"
        opponent_side += str(state[OFFSET_STAT_BOOSTS + NUM_STAT_BOOSTS + i]) + "|"
    for i in range(NUM_WEATHER):
        assert(state[OFFSET_WEATHER + i] == 0 or state[OFFSET_WEATHER + i] == 1)
        if (state[OFFSET_WEATHER + i] == 1):
            if i == 0:
                neutral_side += "no weather "
            neutral_side += WEATHER_LOOKUP[i]

    for i in range(NUM_TERRAIN):
        assert(state[OFFSET_TERRAIN + i] == 0 or state[OFFSET_TERRAIN + i] == 1)
        if (state[OFFSET_TERRAIN + i] == 1):
            neutral_side += TERRLIST[i]
    our_side += "hazards:"
    opponent_side += "hazards:"
    for i in range(NUM_HAZARDS):
        assert(state[OFFSET_HAZARDS + i] == 0 or state[OFFSET_HAZARDS + i] == 1)
        if state[OFFSET_HAZARDS + i] == 1:
            our_side += HAZARD_DICT2[i]
        assert(state[OFFSET_HAZARDS + NUM_HAZARDS + i] == 0 or state[OFFSET_HAZARDS + NUM_HAZARDS + i] == 1)
        if state[OFFSET_HAZARDS + NUM_HAZARDS + i] == 1:
            opponent_side += HAZARD_DICT2[i]

    our_side += "|"
    opponent_side += "|"
    for i in range(TEAM_SIZE):
        our_side += str(state[OFFSET_ITEM + i])
        opponent_side += str(state[OFFSET_ITEM + TEAM_SIZE + i])
    return our_side + "\n" + opponent_side + "\n" + neutral_side + "\n"


