package com.service.rulebasedalgorithm;

import java.io.IOException;
import java.text.ParseException;
import java.util.*;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

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
                    ObjectNode obj = rbaService.proxyObject(jsonNode, i);
                    arrayNode.add(obj);
                }

                proxyLogId = (long) i+1;
            }

            HttpGet httpgetEmployees = new HttpGet("http://employee-service:8082/api/v1/employees");
            CloseableHttpResponse responseEmployees =  httpclient.execute(httpgetEmployees);
            String jsonContentEmployee = EntityUtils.toString(responseEmployees.getEntity(), "UTF-8");
            JsonNode jsonNodeEmployee = objectMapper.readTree(jsonContentEmployee);

            HttpPost httpPostToML = new HttpPost("http://ml-service:5000/example_proxy");
            String arrayNodeString = arrayNode.toString();
            StringEntity stringEntity2 = new StringEntity(arrayNodeString);
            httpPostToML.setEntity(stringEntity2);

             // call ML endpoint for proxy log and add anomalies to tasks
             try (CloseableHttpResponse response3 =  httpclient.execute(httpPostToML)) {
                if (response3.getEntity() != null) {
                    String jsonContentML = EntityUtils.toString(response3.getEntity(), "UTF-8");
                    JsonNode jsonNodeML = objectMapper.readTree(jsonContentML).get("message");
                    String actualStringML = jsonNodeML.textValue();
                    JsonNode actualNodeML = objectMapper.readTree(actualStringML);

                    for (JsonNode item : actualNodeML) {
                        ObjectNode obj = rbaService.makeCase7RequestBody(item, jsonNodeEmployee);
                        obj.put("logId", "Proxy" + item.path("id").asInt());
                        String task = obj.toString();
                        tasks.add(task);
                    }
                    
                } else {
                    System.out.println("something went wrong here");
                }
            } catch (IOException e) {
                e.getMessage();
                System.out.println(e.getMessage());
            }

            // add case 6 data to tasks
            for (JsonNode taskPrePost : tasksPrePost) {
                ObjectNode obj = rbaService.makeCase6RequestBody(taskPrePost, jsonNodeEmployee);
                obj.put("logId", "Proxy_" + taskPrePost.path("id").asInt());
                String task = obj.toString();
                tasks.add(task);
            }
            tasksPrePost.clear();

            // post tasks to database
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

            // update suspect column in proxy log
            // HttpPut httpPutSuspectCol = new HttpPut("http://proxy-log:8086/api/v1/suspectUpdate");
            // httpPutSuspectCol.setEntity(stringEntity);
            
            // try (CloseableHttpResponse response2 =  httpclient.execute(httpPutSuspectCol)) {
            //     if (response2.getEntity() != null) {
            //         //System.out.println("all good");
            //     } else {
            //         System.out.println("something went wrong here");
            //     }
            // } catch (IOException e) {
            //     e.getMessage();
            // }
            tasks.clear();

            httpclient.close();
            return ResponseEntity.ok("Complete");
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Not complete");
        }
    }

    @GetMapping("/checkBuildingAccess")
    public ResponseEntity<?> checkBuildingAccess() throws ClientProtocolException, IOException, ParseException {

        objectMapper.findAndRegisterModules();
        List<String> tasks = new ArrayList<>();
        List<JsonNode> tasksPrePost4 = new ArrayList<>();
        List<JsonNode> tasksPrePost5 = new ArrayList<>();
        List<JsonNode> tasksPrePost3 = new ArrayList<>();
        ArrayNode arrayNode = objectMapper.createArrayNode();

        CloseableHttpClient httpclient = HttpClients.createDefault();
        HttpGet httpgetLog = new HttpGet("http://building-access:8087/api/v1/buildingaccesslogs/" + buildingAccessId);
        CloseableHttpResponse responseBodyLog = httpclient.execute(httpgetLog);

        HttpGet httpgetEmployees = new HttpGet("http://employee-service:8082/api/v1/employees");
        CloseableHttpResponse responseEmployees =  httpclient.execute(httpgetEmployees);

        if (responseBodyLog.getEntity() != null && responseEmployees.getEntity() != null) {
            String jsonContent = EntityUtils.toString(responseBodyLog.getEntity(), "UTF-8");
            JsonNode jsonNode = objectMapper.readTree(jsonContent);

            String jsonContentEmployee = EntityUtils.toString(responseEmployees.getEntity(), "UTF-8");
            JsonNode jsonNodeEmployee = objectMapper.readTree(jsonContentEmployee);

            for (int i = 0; i < jsonNode.size(); i++) {
                int employeeId = jsonNode.get(i).path("userId").asInt();

                boolean taskToCheck3 = rbaService.checkCaseThree(jsonNodeEmployee.get(employeeId - 1).path("terminatedDate").asText());
                if (taskToCheck3) {
                    tasksPrePost3.add(jsonNode.get(i));
                    continue;
                } 

                boolean taskToCheck = rbaService.checkCaseFour(jsonNode.get(i).path("status").asText());
                if (taskToCheck) {
                    tasksPrePost4.add(jsonNode.get(i));
                    continue;
                }

                boolean taskToCheck2 = rbaService.checkCaseFive(jsonNode.get(i).path("officeLocation").asText(), jsonNodeEmployee.get(employeeId - 1).path("location").asText());
                if (taskToCheck2) {
                    tasksPrePost5.add(jsonNode.get(i));
                    continue;
                } 
                
                if (!taskToCheck && !taskToCheck2 && !taskToCheck3) {
                    ObjectNode obj = rbaService.buildingObject(jsonNode, i, jsonNodeEmployee, employeeId);
                    arrayNode.add(obj);
                }

                buildingAccessId = (long) i+1;
            }

            HttpPost httpPostToML = new HttpPost("http://ml-service:5000/example_building");
            String arrayNodeString = arrayNode.toString();
            StringEntity stringEntity2 = new StringEntity(arrayNodeString);
            httpPostToML.setEntity(stringEntity2);

            // call ML endpoint for building access and add anomalies to tasks
            try (CloseableHttpResponse response3 =  httpclient.execute(httpPostToML)) {
                if (response3.getEntity() != null) {
                    String jsonContentML = EntityUtils.toString(response3.getEntity(), "UTF-8");
                    JsonNode jsonNodeML = objectMapper.readTree(jsonContentML).get("message");
                    String actualStringML = jsonNodeML.textValue();
                    JsonNode actualNodeML = objectMapper.readTree(actualStringML);

                    for (JsonNode item : actualNodeML) {
                        ObjectNode obj = rbaService.makeCase7RequestBody(item, jsonNodeEmployee);
                        obj.put("logId", "Building_" + item.path("id").asInt());
                        String task = obj.toString();
                        tasks.add(task);
                    }
                } else {
                    System.out.println("something went wrong here");
                }
            } catch (IOException e) {
                e.getMessage();
                System.out.println(e.getMessage());
            }

            // add case 3 to tasks
            for (JsonNode taskPrePost : tasksPrePost3) {
                ObjectNode obj = rbaService.makeCase3RequestBody(taskPrePost, jsonNodeEmployee);
                obj.put("logId", "Building_" + taskPrePost.path("id").asInt());
                String task = obj.toString();
                tasks.add(task);
            }
            tasksPrePost3.clear();
            
            // add case 4 to tasks
            for (JsonNode taskPrePost : tasksPrePost4) {
                ObjectNode obj = rbaService.makeCase4RequestBody(taskPrePost, jsonNodeEmployee);
                obj.put("logId", "Building_" + taskPrePost.path("id").asInt());
                String task = obj.toString();
                tasks.add(task);
            }
            tasksPrePost4.clear();

            // add case 5 to tasks
            for (JsonNode taskPrePost : tasksPrePost5) {
                ObjectNode obj = rbaService.makeCase5RequestBody(taskPrePost, jsonNodeEmployee);
                obj.put("logId", "Building_" + taskPrePost.path("id").asInt());
                String task = obj.toString();
                tasks.add(task);
            }
            tasksPrePost5.clear();

            // post tasks to database
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

            // update suspect column in building access
            // HttpPut httpPutSuspectCol = new HttpPut("http://building-access:8087/api/v1/suspectUpdate");
            // httpPutSuspectCol.setEntity(stringEntity);
            
            // try (CloseableHttpResponse response2 =  httpclient.execute(httpPutSuspectCol)) {
            //     if (response2.getEntity() != null) {
            //         //System.out.println("all good");
            //     } else {
            //         System.out.println("something went wrong here");
            //     }
            // } catch (IOException e) {
            //     e.getMessage();
            // }
            tasks.clear();

            httpclient.close();
            return ResponseEntity.ok("Complete");
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Not complete");
        }
        
    }

    @GetMapping("/checkPcAccess") 
    public ResponseEntity<?> checkPcAccess() throws ClientProtocolException, IOException, ParseException {

        objectMapper.findAndRegisterModules();
        List<String> tasks = new ArrayList<>();
        List<JsonNode> tasksPrePost1 = new ArrayList<>();
        List<JsonNode> tasksPrePost2 = new ArrayList<>();
        List<JsonNode> tasksPrePost3 = new ArrayList<>();
        List<JsonNode> tasksPrePost5 = new ArrayList<>();
        ArrayNode arrayNode = objectMapper.createArrayNode();

        CloseableHttpClient httpclient = HttpClients.createDefault();
        HttpGet httpgetLog = new HttpGet("http://pc-access:8088/api/v1/pcaccesslogs/" + pcAccessId);
        CloseableHttpResponse responseBodyLog = httpclient.execute(httpgetLog);

        HttpGet httpgetEmployees = new HttpGet("http://employee-service:8082/api/v1/employees");
        CloseableHttpResponse responseEmployees =  httpclient.execute(httpgetEmployees);

        HttpGet httpgetBALog = new HttpGet("http://building-access:8087/api/v1/buildingaccesslogs");
        CloseableHttpResponse responseBodyBALog = httpclient.execute(httpgetBALog);
        
        if (responseBodyLog.getEntity() != null) {
            String jsonContent = EntityUtils.toString(responseBodyLog.getEntity(), "UTF-8");
            JsonNode jsonNode = objectMapper.readTree(jsonContent);

            String jsonContentEmployee = EntityUtils.toString(responseEmployees.getEntity(), "UTF-8");
            JsonNode jsonNodeEmployee = objectMapper.readTree(jsonContentEmployee);

            String jsonContentBA = EntityUtils.toString(responseBodyBALog.getEntity(), "UTF-8");
            JsonNode jsonNodeBA = objectMapper.readTree(jsonContentBA);

            for (int i = 0; i < jsonNode.size(); i++) {
                int employeeId = jsonNode.get(i).path("userId").asInt();

                int taskToCheck2 = rbaService.checkCaseTwo(jsonNode.get(i).path("accessDateTime").asText(), jsonNode.get(i).path("userId").asInt(), jsonNode.get(i).path("machineLocation").asText(), jsonNodeEmployee.get(employeeId - 1).path("location").asText(), jsonNodeBA);
                if (taskToCheck2 == 2) {
                    tasksPrePost2.add(jsonNode.get(i));
                    continue;
                } else if (taskToCheck2 == 5) {
                    tasksPrePost5.add(jsonNode.get(i));
                    continue;
                }

                boolean taskToCheck = rbaService.checkCaseThree(jsonNodeEmployee.get(employeeId - 1).path("terminatedDate").asText());
                if (taskToCheck) {
                    tasksPrePost3.add(jsonNode.get(i));
                    continue;
                } 

                boolean taskToCheck3 = rbaService.checkCaseOne(jsonNode.get(i).path("accessDateTime").asText(), jsonNode.get(i).path("workingHours").asInt());
                if (taskToCheck3) {
                    tasksPrePost1.add(jsonNode.get(i));
                    if (jsonNode.get(i).path("logOnOff").asText().equals("Log On")) {
                        tasksPrePost1.add(jsonNode.get(i+1));
                        i+=1;
                        continue;
                    } else {
                        tasksPrePost1.add(jsonNode.get(i-1));
                        continue;
                    }
                }

                if (!taskToCheck && taskToCheck2 == 0 && !taskToCheck3) {
                    ObjectNode obj = rbaService.pcObject(jsonNode, jsonNodeEmployee, employeeId, i);
                    arrayNode.add(obj);
                }
                
                pcAccessId = (long) i+1;
            }

            // call ML endpoint for pc access and add anomalies to tasks
            HttpPost httpPostToML = new HttpPost("http://ml-service:5000/example_pc");
            String arrayNodeString = arrayNode.toString();
            StringEntity stringEntity2 = new StringEntity(arrayNodeString);
            httpPostToML.setEntity(stringEntity2);

            try (CloseableHttpResponse response3 =  httpclient.execute(httpPostToML)) {
                if (response3.getEntity() != null) {
                    String jsonContentML = EntityUtils.toString(response3.getEntity(), "UTF-8");
                    JsonNode jsonNodeML = objectMapper.readTree(jsonContentML).get("message");
                    String actualStringML = jsonNodeML.textValue();
                    JsonNode actualNodeML = objectMapper.readTree(actualStringML);

                    for (JsonNode item : actualNodeML) {
                        ObjectNode obj = rbaService.makeCase7RequestBody(item, jsonNodeEmployee);
                        obj.put("logId", "PC_" + item.path("id").asInt());
                        String task = obj.toString();
                        tasks.add(task);
                    }
                } else {
                    System.out.println("something went wrong here");
                }
            } catch (IOException e) {
                e.getMessage();
                System.out.println(e.getMessage());
            }

            // add case 3 to tasks
            for (JsonNode taskPrePost : tasksPrePost3) {
                ObjectNode obj = rbaService.makeCase3RequestBody(taskPrePost, jsonNodeEmployee);
                obj.put("logId", "PC_" + taskPrePost.path("id").asInt());
                String task = obj.toString();
                tasks.add(task);
            }
            tasksPrePost3.clear();

            // add case 1 to tasks
            for (JsonNode taskPrePost : tasksPrePost1) {
                ObjectNode obj = rbaService.makeCase1RequestBody(taskPrePost, jsonNodeEmployee);
                obj.put("logId", "PC_" + taskPrePost.path("id").asInt());
                String task = obj.toString();
                tasks.add(task);
            }
            tasksPrePost1.clear();

            // add case 2 to tasks
            for (JsonNode taskPrePost : tasksPrePost2) {
                ObjectNode obj = rbaService.makeCase2RequestBody(taskPrePost, jsonNodeEmployee);
                obj.put("logId", "PC_" + taskPrePost.path("id").asInt());
                String task = obj.toString();
                tasks.add(task);
            }
            tasksPrePost2.clear();

            // add case 5 to tasks
            for (JsonNode taskPrePost : tasksPrePost5) {
                ObjectNode obj = rbaService.makeCase5RequestBody(taskPrePost, jsonNodeEmployee);
                obj.put("logId", "PC_" + taskPrePost.path("id").asInt());
                String task = obj.toString();
                tasks.add(task);
            }
            tasksPrePost5.clear();
            
            // post tasks to database
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

            // update suspect column in pc access
            // HttpPut httpPutSuspectCol = new HttpPut("http://pc-access:8088/api/v1/suspectUpdate");
            // httpPutSuspectCol.setEntity(stringEntity);
            
            // try (CloseableHttpResponse response2 =  httpclient.execute(httpPutSuspectCol)) {
            //     if (response2.getEntity() != null) {
            //         //System.out.println("all good");
            //     } else {
            //         System.out.println("something went wrong here");
            //     }
            // } catch (IOException e) {
            //     e.getMessage();
            // }
            tasks.clear();

            httpclient.close();
            return ResponseEntity.ok("Complete");

        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Not complete");
        }
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

            //System.out.println(jsonNode);

            for (int i = 0; i < jsonNode.size(); i++) {
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
                        // update suspected cases
                        //System.out.println("case update");
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
            
            try (CloseableHttpResponse response =  httpclient.execute(httpPostBA)) {
                if (response.getEntity() != null) {
                    //System.out.println("all good");
                } else {
                    System.out.println("something went wrong here");
                }
            } catch (IOException e) {
                e.getMessage();
            }
            bARows.clear();
            baMap.clear();

            httpclient.close();
            return ResponseEntity.ok("Complete");
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Not complete");
        }
    }
}
