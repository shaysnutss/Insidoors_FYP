package com.service.rulebasedalgorithm;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.ParseException;
import java.util.*;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.opencsv.CSVWriter;

import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.client.methods.HttpPut;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping(path = "/api/v1")
public class RuleBasedAlgorithmController {

    ObjectMapper objectMapper = new ObjectMapper();
    @Autowired private RuleBasedAlgorithmService rbaService;
    Long proxyLogId = 0L;
    Long pcAccessId = 0L;
    Long buildingAccessId = 0L;
    Long taskId = 0L;
    Map<Integer, Integer> baMap = new HashMap<>();

    @GetMapping("/checkProxyLog")
    public ResponseEntity<?> checkProxyLog() throws ClientProtocolException, IOException {

        objectMapper.findAndRegisterModules();
        List<String> tasks = new ArrayList<>();
        List<JsonNode> tasksPrePost = new ArrayList<>();
        ArrayNode arrayNode = objectMapper.createArrayNode();

        CloseableHttpClient httpclient = HttpClients.createDefault();
        HttpGet httpgetLog = new HttpGet("http://proxy-log:8086/api/v1/proxylogs/" + proxyLogId);
        CloseableHttpResponse responseBodyLog = httpclient.execute(httpgetLog);

        if (responseBodyLog.getEntity() != null) {
            String jsonContent = EntityUtils.toString(responseBodyLog.getEntity(), "UTF-8");
            JsonNode jsonNode = objectMapper.readTree(jsonContent);

            for (int i = 0; i < jsonNode.size(); i++) {

                boolean taskToCheck = rbaService.checkCaseSix(jsonNode.get(i).path("bytesOut").asInt());
                if (taskToCheck) {
                    tasksPrePost.add(jsonNode.get(i));
                } else {
                    arrayNode.add(jsonNode.get(i));
                }

                proxyLogId = (long) i+1;

            }

            HttpGet httpgetEmployees = new HttpGet("http://employee-service:8082/api/v1/employees");
            CloseableHttpResponse responseEmployees =  httpclient.execute(httpgetEmployees);
            String jsonContentEmployee = EntityUtils.toString(responseEmployees.getEntity(), "UTF-8");
            JsonNode jsonNodeEmployee = objectMapper.readTree(jsonContentEmployee);

            for (JsonNode taskPrePost : tasksPrePost) {
                ObjectNode obj = objectMapper.createObjectNode();
            
                // System.out.println("for each loop id: " + taskPrePost.path("id").asInt());
                // System.out.println("for each loop employee id: " + taskPrePost.path("userId").asInt());
                obj.put("id", taskPrePost.path("id").asInt());
                obj.put("incidentDesc", jsonNodeEmployee.get(taskPrePost.path("userId").asInt() - 1).path("firstname").asText() + " " + jsonNodeEmployee.get(taskPrePost.path("userId").asInt() - 1).path("lastname").asText() + " has made an unusually large amount of data upload or download.");
                obj.put("incidentTitle", "Potential Data Exfiltration");
                obj.put("severity", 100);
                obj.put("accountId", 0);
                obj.put("employeeId", taskPrePost.path("userId").asInt());
                obj.put("suspect", 6);
                String task = obj.toString();
                tasks.add(task);
            }

            HttpPost httpPostTask = new HttpPost("http://task-management-service:8081/api/v1/tasks");
            String combinedString = String.join("/", tasks);
            StringEntity stringEntity = new StringEntity(combinedString);
            httpPostTask.setEntity(stringEntity);
            
            try (CloseableHttpResponse response =  httpclient.execute(httpPostTask)) {

                if (response.getEntity() != null) {
                    //System.out.println("all good");
                } else {
                    System.out.println("something went wrong here");
                }

            } catch (IOException e) {
                e.getMessage();
            }

            HttpPut httpPutSuspectCol = new HttpPut("http://proxy-log:8086/api/v1/suspectUpdate");
            httpPutSuspectCol.setEntity(stringEntity);
            
            try (CloseableHttpResponse response2 =  httpclient.execute(httpPutSuspectCol)) {

                if (response2.getEntity() != null) {
                    //System.out.println("all good");
                } else {
                    System.out.println("something went wrong here");
                }

            } catch (IOException e) {
                e.getMessage();
            }

            HttpPost httpPostToML = new HttpPost("http://ml-service:5000/example");
            StringEntity stringEntity2 = new StringEntity(arrayNode.toString());
            httpPostToML.setEntity(stringEntity2);

            try (CloseableHttpResponse response3 =  httpclient.execute(httpPostToML)) {

                if (response3.getEntity() != null) {
                    String jsonML = EntityUtils.toString(response3.getEntity(), "UTF-8");
                    JsonNode jsonToRead = objectMapper.readTree(jsonML);

                    for (int l = 0; l < jsonToRead.size(); l++) {
                        System.out.println(jsonToRead.get(l));
                    }
                } else {
                    System.out.println("something went wrong here");
                }

            } catch (IOException e) {
                e.getMessage();
                System.out.println(e.getMessage());
            }

            return ResponseEntity.ok("Complete");
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Not complete");
        }
    }

    @GetMapping("/checkBuildingAccess")
    public ResponseEntity<?> checkBuildingAccess() throws ClientProtocolException, IOException, ParseException {

        objectMapper.findAndRegisterModules();
        List<String> tasks = new ArrayList<>();
        List<String[]> csvData = new ArrayList<>();

        CloseableHttpClient httpclient = HttpClients.createDefault();
        HttpGet httpgetLog = new HttpGet("http://building-access:8087/api/v1/buildingaccesslogs/" + buildingAccessId);
        CloseableHttpResponse responseBodyLog = httpclient.execute(httpgetLog);

        if (responseBodyLog.getEntity() != null) {
            String jsonContent = EntityUtils.toString(responseBodyLog.getEntity(), "UTF-8");
            JsonNode jsonNode = objectMapper.readTree(jsonContent);

            for (int i = 0; i < jsonNode.size(); i++) {
                String idArray = jsonNode.get(i).path("id").asText();
                String userIdArray = jsonNode.get(i).path("userId").asText();
                String accessDateTime = jsonNode.get(i).path("accessDateTime").asText();
                String direction = jsonNode.get(i).path("direction").asText();
                String status = jsonNode.get(i).path("status").asText();
                String officeLocation = jsonNode.get(i).path("officeLocation").asText();
                String suspect = jsonNode.get(i).path("suspect").asText();

                String[] csvArray = {idArray, userIdArray, accessDateTime, direction, status, officeLocation, suspect};

                String taskToCheck = rbaService.checkCaseFour(jsonNode.get(i).path("status").asText(), jsonNode.get(i).path("userId").asInt());
                if (!taskToCheck.equals("")) {
                    tasks.add(taskToCheck);
                    System.out.println("Case 4"+jsonNode.get(i).path("id").asText());
                } 

                String taskToCheck2 = rbaService.checkCaseFive(jsonNode.get(i).path("userId").asInt(), jsonNode.get(i).path("officeLocation").asText());
                if (!taskToCheck2.equals("")) {
                    tasks.add(taskToCheck2);
                    System.out.println("Case 5"+jsonNode.get(i).path("id").asText());
                } 
                
                if (taskToCheck.equals("") && taskToCheck2.equals("")) {
                    csvData.add(csvArray);
                    //writer.writeNext(csvArray);
                }

                buildingAccessId = (long) i+1;
            }

            HttpPost httpPostTask = new HttpPost("http://task-management-service:8081/api/v1/tasks");
            String combinedString = String.join("/", tasks);
            StringEntity stringEntity = new StringEntity(combinedString);
            httpPostTask.setEntity(stringEntity);

            try (CloseableHttpResponse response =  httpclient.execute(httpPostTask)) {

                if (response.getEntity() != null) {
                    //System.out.println("all good");
                } else {
                    System.out.println("something went wrong here");
                }

            } catch (IOException e) {
                e.getMessage();
            }

            return ResponseEntity.ok("Complete");
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Not complete");
        }
        
    }

    @GetMapping("/checkPcAccess") 
    public ResponseEntity<?> checkPcAccess() throws ClientProtocolException, IOException, ParseException {

        objectMapper.findAndRegisterModules();
        List<String> tasks = new ArrayList<>();
        List<String[]> csvData = new ArrayList<>();

        CloseableHttpClient httpclient = HttpClients.createDefault();
        HttpGet httpgetLog = new HttpGet("http://pc-access:8088/api/v1/pcaccesslogs/" + pcAccessId);
        CloseableHttpResponse responseBodyLog = httpclient.execute(httpgetLog);
        
        if (responseBodyLog.getEntity() != null) {
            String jsonContent = EntityUtils.toString(responseBodyLog.getEntity(), "UTF-8");
            JsonNode jsonNode = objectMapper.readTree(jsonContent);

            for (int i = 0; i < jsonNode.size(); i++) {
                String idArray = jsonNode.get(i).path("id").asText();
                String userIdArray = jsonNode.get(i).path("userId").asText();
                String accessDateTime = jsonNode.get(i).path("accessDateTime").asText();
                String logOnOff = jsonNode.get(i).path("logOnOff").asText();
                String machineName = jsonNode.get(i).path("machineName").asText();
                String machineLocation = jsonNode.get(i).path("machineLocation").asText();
                String suspect = jsonNode.get(i).path("suspect").asText();
                String workingHours = jsonNode.get(i).path("workingHours").asText();

                String[] csvArray = {idArray, userIdArray, accessDateTime, logOnOff, machineName, machineLocation, suspect, workingHours};

                String taskToCheck2 = rbaService.checkCaseTwo(jsonNode.get(i).path("accessDateTime").asText(), jsonNode.get(i).path("userId").asInt(), jsonNode.get(i).path("machineLocation").asText());
                if (taskToCheck2.equals("case 5")) {
                    continue;
                } 

                if (!taskToCheck2.equals("")) {
                    tasks.add(taskToCheck2);
                    System.out.println("Adding to task Case 2: " + jsonNode.get(i).path("id").asInt());
                    continue;
                } 

                String taskToCheck = rbaService.checkCaseThree(jsonNode.get(i).path("logOnOff").asText(), jsonNode.get(i).path("userId").asInt());
                if (!taskToCheck.equals("")) {
                    tasks.add(taskToCheck);
                    System.out.println("Adding to task Case 3");
                    continue;
                } 

                String taskToCheck3 = rbaService.checkCaseOne(jsonNode.get(i).path("accessDateTime").asText(), jsonNode.get(i).path("userId").asInt(), jsonNode.get(i).path("workingHours").asInt());
                if (!taskToCheck3.equals("")) {
                    tasks.add(taskToCheck3);
                    System.out.println("Adding to task Case 1: " + jsonNode.get(i).path("id").asInt());
                    if (jsonNode.get(i).path("logOnOff").asText().equals("Log On")) {
                        String taskToCheckNext = rbaService.checkCaseOne(jsonNode.get(i+1).path("accessDateTime").asText(), jsonNode.get(i+1).path("userId").asInt(), jsonNode.get(i+1).path("workingHours").asInt());
                        tasks.add(taskToCheckNext);
                        System.out.println("Adding to task Case 1: " + jsonNode.get(i+1).path("id").asInt());
                        i+=1;
                        continue;
                    } else {
                        String taskToCheckPrev = rbaService.checkCaseOne(jsonNode.get(i-1).path("accessDateTime").asText(), jsonNode.get(i-1).path("userId").asInt(), jsonNode.get(i-1).path("workingHours").asInt());
                        tasks.add(taskToCheckPrev);
                        System.out.println("Adding to task Case 1: " + jsonNode.get(i-1).path("id").asInt());
                        continue;
                    }
                    
                } 

                if (taskToCheck.equals("") && taskToCheck2.equals("") && taskToCheck3.equals("")) {
                    csvData.add(csvArray);
                }

                pcAccessId = (long) i+1;
               

            }
            
            HttpPost httpPostTask = new HttpPost("http://task-management-service:8081/api/v1/tasks");
            String combinedString = String.join("/", tasks);
            StringEntity stringEntity = new StringEntity(combinedString);
            httpPostTask.setEntity(stringEntity);
            //System.out.println(combinedString);
           
            try (CloseableHttpResponse response =  httpclient.execute(httpPostTask)) {

                if (response.getEntity() != null) {
                    System.out.println("all good");
                    //System.out.println(response.getEntity().getContentLength());
                } else {
                    System.out.println("something went wrong here");
                }

            } catch (IOException e) {
                e.getMessage();
            }

            return ResponseEntity.ok("Complete");

        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Not complete");
        }
    }

    public boolean idExists(int id) throws org.apache.http.ParseException, IOException {

        CloseableHttpClient httpclient = HttpClients.createDefault();
        HttpGet httpgetBA = new HttpGet("http://behavioral-analysis-service:8084/api/v1/behavioralanalysis");
        CloseableHttpResponse responseBodyBA = httpclient.execute(httpgetBA);

        if (responseBodyBA.getEntity() != null) {
            String jsonContentBA = EntityUtils.toString(responseBodyBA.getEntity(), "UTF-8");
            JsonNode jsonNodeBA = objectMapper.readTree(jsonContentBA);

            for (int j = 0; j < jsonNodeBA.size(); j++) {
                System.out.println(jsonNodeBA.get(j).path("id").asInt());
                if (id == jsonNodeBA.get(j).path("employeeId").asInt()) {
                    return true;
                }
            }
            System.out.println("-------");
        }
        
        return false;
    }

    @GetMapping("/postToBA") 
    public ResponseEntity<?> postToBA() throws ClientProtocolException, IOException, ParseException {

        objectMapper.findAndRegisterModules();
        List<String> bARows = new ArrayList<>();

        CloseableHttpClient httpclient = HttpClients.createDefault();
        HttpGet httpgetLog = new HttpGet("http://task-management-service:8081/api/v1/tasks/afterId/" + taskId);
        CloseableHttpResponse responseBodyLog = httpclient.execute(httpgetLog);
        
        if (responseBodyLog.getEntity() != null) {

            String jsonContent = EntityUtils.toString(responseBodyLog.getEntity(), "UTF-8");
            JsonNode jsonNode = objectMapper.readTree(jsonContent);

            for (int i = 0; i < jsonNode.size(); i++) {

                System.out.println("employee id from task: " + jsonNode.get(i).path("employeeId").asInt());
                HttpGet httpgetBA = new HttpGet("http://behavioral-analysis-service:8084/api/v1/behavioralanalysis/employee/" + jsonNode.get(i).path("employeeId").asInt());

                try (CloseableHttpResponse responseBodyBA = httpclient.execute(httpgetBA);) {
        
                    String jsonContentBA = EntityUtils.toString(responseBodyBA.getEntity(), "UTF-8");
                    JsonNode jsonNodeBA = objectMapper.readTree(jsonContentBA);

                    if (jsonNodeBA == null || jsonNodeBA.size() == 0) {
                        // create new entry and add to list
                        if (!baMap.containsKey(jsonNode.get(i).path("employeeId").asInt())) {
                            baMap.put(jsonNode.get(i).path("employeeId").asInt(), 1);
                        } else {
                            int cases = baMap.get(jsonNode.get(i).path("employeeId").asInt());
                            baMap.put(jsonNode.get(i).path("employeeId").asInt(), cases+1);
                        }
    
                    } else if (!jsonNodeBA.get("id").isNull()){
                        // then update suspected cases and break
                        System.out.println("case update");
                        HttpPut httpUpdateCases = new HttpPut("http://behavioral-analysis-service:8084/api/v1/updateSuspectedCases/" + jsonNode.get(i).path("employeeId").asInt());
                        String jsonString = "{\"suspectedCases\": 1}";
                        StringEntity stringEntity = new StringEntity(jsonString);
                        httpUpdateCases.setEntity(stringEntity);
                        
                        try (CloseableHttpResponse response =  httpclient.execute(httpUpdateCases)) {
            
                            if (response.getEntity() != null) {
                                //System.out.println("all good");
                            } else {
                                System.out.println("something went wrong here");
                            }
            
                        } catch (IOException e) {
                            e.getMessage();
                        }
                    
                    }
    
                } catch (IOException e) {
                    e.getMessage();
                }

                taskId = (long) i+1;

            }

            for (int key : baMap.keySet()) {
                ObjectMapper objectMapper = new ObjectMapper();
                ObjectNode obj = objectMapper.createObjectNode();

                obj.put("employeeId", key);
                obj.put("suspectedCases", baMap.get(key));
                String newBA = obj.toString();
                bARows.add(newBA);
            }

            HttpPost httpPostBA = new HttpPost("http://behavioral-analysis-service:8084/api/v1/behavioralanalysis");
            String combinedString = String.join("/", bARows);
            StringEntity stringEntity = new StringEntity(combinedString);
            httpPostBA.setEntity(stringEntity);
            System.out.println(combinedString);
            
            try (CloseableHttpResponse response =  httpclient.execute(httpPostBA)) {

                if (response.getEntity() != null) {
                    //System.out.println("all good");
                } else {
                    System.out.println("something went wrong here");
                }

            } catch (IOException e) {
                e.getMessage();
            }

            return ResponseEntity.ok("Complete");
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Not complete");
        }
    }


}
