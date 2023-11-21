package com.service.rulebasedalgorithm;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.ParseException;
import java.util.*;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.opencsv.CSVWriter;

import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
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

    @GetMapping("/checkProxyLog")
    public ResponseEntity<?> checkProxyLog() throws ClientProtocolException, IOException {

        objectMapper.findAndRegisterModules();
        List<String> tasks = new ArrayList<>();
        List<String[]> csvData = new ArrayList<>();

        CloseableHttpClient httpclient = HttpClients.createDefault();
        HttpGet httpgetLog = new HttpGet("http://proxy-log:8086/api/v1/proxylogs/" + proxyLogId);
        CloseableHttpResponse responseBodyLog = httpclient.execute(httpgetLog);

        //File file = new File("app/data/proxy_log.csv"); 

        // String[] header = {"id", "user_id", "access_date_time", "machine_name", "url", "category", "bytes_in", "bytes_out", "suspect"};
        // if (!file.exists()) {
        //     csvData.add(header);
        // }

        if (responseBodyLog.getEntity() != null) {
            String jsonContent = EntityUtils.toString(responseBodyLog.getEntity(), "UTF-8");
            JsonNode jsonNode = objectMapper.readTree(jsonContent);

            for (int i = 0; i < jsonNode.size(); i++) {

                String idArray = jsonNode.get(i).path("id").asText();
                String userIdArray = jsonNode.get(i).path("userId").asText();
                String accessDateTime = jsonNode.get(i).path("accessDateTime").asText();
                String machineName = jsonNode.get(i).path("machineName").asText();
                String url = jsonNode.get(i).path("url").asText();
                String category = jsonNode.get(i).path("category").asText();
                String bytesIn = jsonNode.get(i).path("bytesIn").asText();
                String bytesOut = jsonNode.get(i).path("bytesOut").asText();
                String suspect = jsonNode.get(i).path("suspect").asText();

                String[] csvArray = {idArray, userIdArray, accessDateTime, machineName, url, category, bytesIn, bytesOut, suspect};

                String taskToCheck = rbaService.checkCaseSix(jsonNode.get(i).path("bytesOut").asInt(), jsonNode.get(i).path("userId").asInt());
                if (!taskToCheck.equals("")) {
                    tasks.add(taskToCheck);
                    System.out.println(jsonNode.get(i).path("id").asText());
                } else {
                    csvData.add(csvArray);
                }

                proxyLogId = (long) i+1;

            }

            // try {
            //     FileWriter outputfile = new FileWriter(file, true);
            //     CSVWriter writer = new CSVWriter(outputfile);
            //     writer.writeAll(csvData);
            //     writer.flush();
            //     System.out.println("all entered");
            //     writer.close();
            // } catch (IOException e) {
            //     e.printStackTrace();
            // }

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
            //System.out.println("in if statement before loop");
            // pcAccessId = (long) jsonNode.size();

            // Iterator<JsonNode> iterator = jsonNode.elements();
            // boolean skipNextIteration = false;
            // JsonNode previousNode = null;

            // while (iterator.hasNext()) {
            //     JsonNode currentNode = iterator.next();

            //     if (skipNextIteration) {
            //         // Skip the current iteration
            //         skipNextIteration = false;
            //         continue;
            //     }

            //     String idArray = currentNode.path("id").asText();
            //     String userIdArray = currentNode.path("userId").asText();
            //     String accessDateTime = currentNode.path("accessDateTime").asText();
            //     String logOnOff = currentNode.path("logOnOff").asText();
            //     String machineName = currentNode.path("machineName").asText();
            //     String machineLocation = currentNode.path("machineLocation").asText();
            //     String suspect = currentNode.path("suspect").asText();
            //     String workingHours = currentNode.path("workingHours").asText();

            //     String[] csvArray = {idArray, userIdArray, accessDateTime, logOnOff, machineName, machineLocation, suspect, workingHours};

            //     String taskToCheck2 = rbaService.checkCaseTwo(currentNode.path("accessDateTime").asText(), currentNode.path("userId").asInt(), currentNode.path("machineLocation").asText());
            //     if (taskToCheck2.equals("case 5")) {
            //         iterator.remove();
            //         //System.out.println("Removing case 5: "+currentNode.path("id").asText());
            //         continue;
            //     }

            //     if (!taskToCheck2.equals("")) {
            //         tasks.add(taskToCheck2);
            //         iterator.remove();
            //         //System.out.println("Removing case 2: "+currentNode.path("id").asText());
            //         continue;                    
            //     } 

            //     String taskToCheck = rbaService.checkCaseThree(currentNode.path("logOnOff").asText(), currentNode.path("userId").asInt());
            //     if (!taskToCheck.equals("")) {
            //         tasks.add(taskToCheck);
            //         iterator.remove();
            //         //System.out.println("Removing case 3: "+currentNode.path("id").asText());
            //         continue;
            //     } 

            //     String taskToCheck3 = rbaService.checkCaseOne(currentNode.path("accessDateTime").asText(), currentNode.path("userId").asInt(), currentNode.path("workingHours").asInt());
            //     if (!taskToCheck3.equals("")) {
            //         tasks.add(taskToCheck3);
            //         //System.out.println("Case 1: "+ currentNode.path("id").asText());
            //         if (iterator.hasNext() && currentNode.path("logOnOff").asText().equals("Log On")) {
            //             JsonNode nextNode = iterator.next();
            //             String taskToCheckNext = rbaService.checkCaseOne(nextNode.path("accessDateTime").asText(), nextNode.path("userId").asInt(), nextNode.path("workingHours").asInt());
            //             tasks.add(taskToCheckNext);
            //             //System.out.println("Case 1: "+ nextNode.path("id").asText());
            
            //             // Skip the next iteration
            //             skipNextIteration = true;
            //         } else {
            //             JsonNode prevNode = previousNode;
            //             String taskToCheckPrev = rbaService.checkCaseOne(prevNode.path("accessDateTime").asText(), prevNode.path("userId").asInt(), prevNode.path("workingHours").asInt());
            //             tasks.add(taskToCheckPrev);
            //             //System.out.println("Case 1: "+ prevNode.path("id").asText());
            
            //         }
            //     } 

            //     if (taskToCheck.equals("") && taskToCheck2.equals("") && taskToCheck3.equals("")) {
            //         csvData.add(csvArray);
            //     }

            //     previousNode = currentNode;
            // }


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

            // String combinedString = "";
            // for (int j = 0; i < tasks.size(); i++) {
            //     System.out.println(tasks.get(j).toString().trim());
            //     combinedString += tasks.get(j).toString();
            //     if (i != tasks.size()-1) {
            //         combinedString += "/";
            //     }
            // }
            
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

}
