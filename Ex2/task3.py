
old_scenario=open("scenarios/New_scenario.scenario", "r")


# print(content_old_scenario)

new_scenario = open("scenarios/Modified_New_scenario.scenario", "w+")
i=0
for line in old_scenario:

    if "dynamicElements" in line:

        new_scenario.write(line[0:27]+""" {
  \"source\" : null,
  \"targetIds\" : [ 2 ],
  \"position\" : {
    \"x\" : 1.5,
    \"y\" : 1.8
  },
  \"velocity\" : {
    \"x\" : 0.0,
    \"y\" : 0.0
  },
  \"nextTargetListIndex\" : 0,
  \"freeFlowSpeed\" : 1.2399599214195676,
  \"attributes\" : {
    \"id\" : -1,
    \"radius\" : 0.2,
    \"densityDependentSpeed\" : false,
    \"speedDistributionMean\" : 1.34,
    \"speedDistributionStandardDeviation\" : 0.26,
    \"minimumSpeed\" : 0.5,
    \"maximumSpeed\" : 2.2,
    \"acceleration\" : 2.0,
    \"footStepsToStore\" : 4,
    \"searchRadius\" : 1.0,
    \"angleCalculationType\" : \"USE_CENTER\",
    \"targetOrientationAngleThreshold\" : 45.0
  },
  \"idAsTarget\" : -1,
  \"isChild\" : false,
  \"isLikelyInjured\" : false,
  \"mostImportantEvent\" : null,
  \"salientBehavior\" : \"TARGET_ORIENTED\",
  \"groupIds\" : [ ],
  \"trajectory\" : {
    \"footSteps\" : [ ]
  },
  \"groupSizes\" : [ ],
  \"modelPedestrianMap\" : null,
  \"type\" : \"PEDESTRIAN\"
} """+line[27:])
        print(line[27:])

    else:

        new_scenario.write(line)



new_scenario.close()
old_scenario.close()
