import json
import os
import numpy as np

NumofAgents=15
err = np.random.normal(loc = 0, scale = 1, size = (2, NumofAgents))
#adding intial position for osm here
postvis=open("/home/navid/Ex2/output/Modified_Correct_scenario.scenario_2019-11-12_17-56-14.450/postvis.traj", "r")
index=0
data = []

for line in postvis:
    if index != 0:
        token = line.split(" ")
        agentID = int(token[0])
        simTime = float(token[1])
        endTime = float(token[2])
        startX = float(token[3])
        startY = float(token[4])
        endX = float(token[5])
        endY = float(token[6])
        velX = (endX - startX )/(endTime-simTime)
        velY = (endY - startY) / (endTime - simTime)
        row = {"agentId" : agentID ,
                "endX"   : endX,
                "endY"   : endY,
                "velX"   : velX,
                "velY"   : velY
               }
        data.append(row)
        # print(row)
    index += 1
arrayTraj = {}
for i in range(NumofAgents):
    arrayTraj[i] = []

for row in data:
    arrayTraj[row["agentId"]].append(row)

#finding maximum timeStep
MaxTimeStep=0
for i in range(NumofAgents):
    if len(arrayTraj[i])> MaxTimeStep:
        MaxTimeStep= len(arrayTraj[i])

print("MaxStepsize"+ str(MaxTimeStep))

for step in range(MaxTimeStep):
    old_scenario = open("/home/navid/Ex2/scenarios/Correct_scenario.scenario", "r")

    # print(content_old_scenario)
    filename = str(step)
    new_scenario = open("/home/navid/Ex2/scenarios/scenario"+filename+".scenario", "w+")

    data = json.load(old_scenario)

    data["name"] = 'scenario' + filename + '.scenario'
    i = 0

    for agentIndex in range(NumofAgents):
        # print("agentIndex"+str(agentIndex))
        # print("step"+str(step))
        if step < len(arrayTraj[agentIndex]):
            # dynamicElements = {
            #     "source": None,
            #     "targetIds": [NumofAgents],
            #     "position": {
            #         "x": arrayTraj[agentIndex][step]["endX"],
            #         "y": arrayTraj[agentIndex][step]["endY"]
            #     },
            #     "velocity": {
            #         "x": arrayTraj[agentIndex][step]["velX"],
            #         "y": arrayTraj[agentIndex][step]["velY"]
            #     },
            #     "nextTargetListIndex": 0,
            #     "freeFlowSpeed": 1.2399599214195676,
            #     "attributes": {
            #         "id": agentIndex,
            #         "radius": 0.2,
            #         "densityDependentSpeed": False,
            #         "speedDistributionMean": 1.34,
            #         "speedDistributionStandardDeviation": 0.26,
            #         "minimumSpeed": 0.5,
            #         "maximumSpeed": 2.2,
            #         "acceleration": 2.0,
            #         "footStepsToStore": 4,
            #         "searchRadius": 1.0,
            #         "angleCalculationType": "USE_CENTER",
            #         "targetOrientationAngleThreshold": 45.0
            #     },
            #     "idAsTarget": -1,
            #     "isChild": False,
            #     "isLikelyInjured": False,
            #     "mostImportantEvent": None,
            #     "salientBehavior": "TARGET_ORIENTED",
            #     "groupIds": [],
            #     "trajectory": {
            #         "footSteps": []
            #     },
            #     "groupSizes": [],
            #     "modelPedestrianMap": None,
            #     "type": "PEDESTRIAN"
            # }
            dynamicElements = {
                  "attributes" : {
                    "id" : agentIndex,
                    "radius" : 0.2,
                    "densityDependentSpeed" : False,
                    "speedDistributionMean" : 1.34,
                    "speedDistributionStandardDeviation" : 0.26,
                    "minimumSpeed" : 0.5,
                    "maximumSpeed" : 2.2,
                    "acceleration" : 2.0,
                    "footstepHistorySize" : 4,
                    "searchRadius" : 1.0,
                    "angleCalculationType" : "USE_CENTER",
                    "targetOrientationAngleThreshold" : 45.0
                  },
                  "source" : None,
                  "targetIds" : [ NumofAgents ],
                  "nextTargetListIndex" : 0,
                  "isCurrentTargetAnAgent" : False,
                  "position" : {
                    "x" : arrayTraj[agentIndex][step]["endX"] + err[0][agentIndex],
                    "y" : arrayTraj[agentIndex][step]["endY"] + err[1][agentIndex]
                  },
                  "velocity" : {
                    "x" : arrayTraj[agentIndex][step]["velX"],
                    "y" : arrayTraj[agentIndex][step]["velY"]
                  },
                  "freeFlowSpeed" : 1.296620673208589,
                  "followers" : [ ],
                  "idAsTarget" : -1,
                  "isChild" : False,
                  "isLikelyInjured" : False,
                  "psychology" : {
                    "mostImportantStimulus" : None,
                    "selfCategory" : "TARGET_ORIENTED"
                  },
                  "groupIds" : [ ],
                  "groupSizes" : [ ],
                  "trajectory" : {
                    "footSteps" : [ ]
                  },
                  "modelPedestrianMap" : None,
                  "type" : "PEDESTRIAN"
                }
            data["scenario"]["topography"]["dynamicElements"].append(dynamicElements)

    # print(data["scenario"]["topography"]["dynamicElements"])


    # print(data["scenario"]["topography"]["dynamicElements"])


    json.dump(data, new_scenario, indent=4)

    new_scenario.close()
    old_scenario.close()
    myCmd = '''/usr/lib/jvm/java-11-openjdk-amd64/bin/java -jar /home/navid/Documents/Praktikum/vadere.master.linux/vadere-console.jar scenario-run --scenario-file "/home/navid/Ex2/scenarios/scenario'''+str(step)+'''.scenario" --output-dir "/home/navid/Ex2/output/results/output'''+str(step)+'''"'''
    print(myCmd)

    os.system(myCmd)


print(arrayTraj[1])


