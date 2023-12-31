package com.service.taskmanagementcomposite;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.client.methods.HttpPut;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.scheduling.annotation.Async;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.HttpServerErrorException;

@RestController
@CrossOrigin(maxAge = 3600)
@RequestMapping(path = "/api/v1")
public class TaskManagementCompositeController {

    // CloseableHttpClient httpclient = HttpClients.createDefault();
    ObjectMapper objectMapper = new ObjectMapper();

    @CrossOrigin(origins = "http://localhost:30008")
    @GetMapping("/viewAllTasks")
    public ResponseEntity<?> viewAllTasks() {
        objectMapper.findAndRegisterModules();
        CloseableHttpClient httpclient = HttpClients.createDefault();
        HttpGet httpgetTask = new HttpGet("http://task-management-service:8081/api/v1/tasks");
        HttpGet httpgetEmployee = new HttpGet("http://employee-service:8082/api/v1/employees");
        HttpGet httpgetAccount = new HttpGet("http://account-service:8080/api/v1/account/getAllAccounts");
        
        try (CloseableHttpResponse responseBodyTask = httpclient.execute(httpgetTask);){
            CloseableHttpResponse responseBodyEmployee = httpclient.execute(httpgetEmployee);
            CloseableHttpResponse responseBodyAccount = httpclient.execute(httpgetAccount);

            if (responseBodyTask.getStatusLine().getStatusCode() == 200) {
                String jsonContentTask = EntityUtils.toString(responseBodyTask.getEntity(), "UTF-8");
                JsonNode jsonNodeTask = objectMapper.readTree(jsonContentTask);

                String jsonContentEmployee = EntityUtils.toString(responseBodyEmployee.getEntity(), "UTF-8");
                JsonNode jsonNodeEmployee = objectMapper.readTree(jsonContentEmployee);

                String jsonContentAccount = EntityUtils.toString(responseBodyAccount.getEntity(), "UTF-8");
                JsonNode jsonNodeAccount = objectMapper.readTree(jsonContentAccount);

                List<Map<String, Object>> taskManagementList = new ArrayList<>();
                for (int i = 0; i < jsonNodeTask.size(); i++) {
                    Map<String, Object> item = new HashMap<>();
                    int employeeId = jsonNodeTask.get(i).path("employeeId").asInt();

                    // task management items
                    item.put("id", jsonNodeTask.get(i).path("id"));
                    item.put("incidentTitle", jsonNodeTask.get(i).path("incidentTitle"));
                    item.put("incidentDesc", jsonNodeTask.get(i).path("incidentDesc"));
                    item.put("incidentTimestamp", jsonNodeTask.get(i).path("incidentTimestamp"));
                    item.put("severity", jsonNodeTask.get(i).path("severity"));
                    item.put("status", jsonNodeTask.get(i).path("status"));
                    item.put("dateAssigned", jsonNodeTask.get(i).path("dateAssigned"));
                    item.put("accountId", jsonNodeTask.get(i).path("accountId"));
                    item.put("truePositive", jsonNodeTask.get(i).path("truePositive"));
                    item.put("logId", jsonNodeTask.get(i).path("logId"));

                    // employee items

                    item.put("employeeFirstname", jsonNodeEmployee.get(employeeId - 1).path("firstname"));
                    item.put("employeeLastname", jsonNodeEmployee.get(employeeId - 1).path("lastname"));

                    // soc account items

                    if (jsonNodeTask.get(i).path("accountId").asInt() == 0) {
                        item.put("socName", "None");
                    } else {
                        item.put("socName", jsonNodeAccount.get(jsonNodeTask.get(i).path("accountId").asInt() - 1).path("name"));
                    }

                    // Add the item to the list
                    taskManagementList.add(item);
                }

                httpclient.close();
                return ResponseEntity.ok(taskManagementList);
            } else {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Task management list not found.");
            }
        } catch (HttpServerErrorException e) {
            return ResponseEntity.status(e.getStatusCode()).body("Error in retrieving task management list.");
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Internal server error.");
        }
    }

    @CrossOrigin(origins = "http://localhost:30008")
    @GetMapping("/viewAllTasksByAccountId/{id}")
    public ResponseEntity<?> viewAllTasksByAccountId(@PathVariable int id) {
        try {
            objectMapper.findAndRegisterModules();
            CloseableHttpClient httpclient = HttpClients.createDefault();

            HttpGet httpgetTask = new HttpGet("http://task-management-service:8081/api/v1/tasks/account/" + id);
            HttpGet httpgetEmployee = new HttpGet("http://employee-service:8082/api/v1/employees");
            HttpGet httpgetAccount = new HttpGet("http://account-service:8080/api/v1/account/getAllAccounts");
            CloseableHttpResponse responseBodyTask = httpclient.execute(httpgetTask);
            CloseableHttpResponse responseBodyEmployee = httpclient.execute(httpgetEmployee);
            CloseableHttpResponse responseBodyAccount = httpclient.execute(httpgetAccount);

            // System.out.println(jsonNodeTask);
            if (responseBodyTask.getStatusLine().getStatusCode() == 200) {
                String jsonContentTask = EntityUtils.toString(responseBodyTask.getEntity(), "UTF-8");
                JsonNode jsonNodeTask = objectMapper.readTree(jsonContentTask);

                String jsonContentEmployee = EntityUtils.toString(responseBodyEmployee.getEntity(), "UTF-8");
                JsonNode jsonNodeEmployee = objectMapper.readTree(jsonContentEmployee);

                String jsonContentAccount = EntityUtils.toString(responseBodyAccount.getEntity(), "UTF-8");
                JsonNode jsonNodeAccount = objectMapper.readTree(jsonContentAccount);

                List<Map<String, Object>> taskManagementList = new ArrayList<>();
                for (int i = 0; i < jsonNodeTask.size(); i++) {
                    Map<String, Object> item = new HashMap<>();
                    int employeeId = jsonNodeTask.get(i).path("employeeId").asInt();

                    // task management items
                    item.put("id", jsonNodeTask.get(i).path("id"));
                    item.put("incidentTitle", jsonNodeTask.get(i).path("incidentTitle"));
                    item.put("incidentDesc", jsonNodeTask.get(i).path("incidentDesc"));
                    item.put("incidentTimestamp", jsonNodeTask.get(i).path("incidentTimestamp"));
                    item.put("severity", jsonNodeTask.get(i).path("severity"));
                    item.put("status", jsonNodeTask.get(i).path("status"));
                    item.put("dateAssigned", jsonNodeTask.get(i).path("dateAssigned"));
                    item.put("accountId", jsonNodeTask.get(i).path("accountId"));
                    item.put("truePositive", jsonNodeTask.get(i).path("truePositive"));
                    item.put("logId", jsonNodeTask.get(i).path("logId"));

                    // employee items
                    item.put("employeeFirstname", jsonNodeEmployee.get(employeeId - 1).path("firstname"));
                    item.put("employeeLastname", jsonNodeEmployee.get(employeeId - 1).path("lastname"));

                    // soc account items
                    if (jsonNodeTask.get(i).path("accountId").asInt() == 0) {
                        item.put("socName", null);
                    } else {
                        item.put("socName", jsonNodeAccount.get(jsonNodeTask.get(i).path("accountId").asInt() - 1).path("name"));
                    }

                    // Add the item to the list
                    taskManagementList.add(item);
                }

                httpclient.close();
                return ResponseEntity.ok(taskManagementList);
            } else {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Task management list not found.");
            }
        } catch (HttpServerErrorException e) {
            return ResponseEntity.status(e.getStatusCode()).body("Error in retrieving task management list.");
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Internal server error.");
        }
    }

    @CrossOrigin(origins = "http://localhost:30008")
    @GetMapping("/viewTaskById/{id}")
    public ResponseEntity<?> viewTaskById(@PathVariable Long id) {
        try {
            objectMapper.findAndRegisterModules();
            CloseableHttpClient httpclient = HttpClients.createDefault();

            HttpGet httpgetTask = new HttpGet("http://task-management-service:8081/api/v1/tasks/" + id);
            HttpGet httpgetEmployee = new HttpGet("http://employee-service:8082/api/v1/employees");
            HttpGet httpgetAccount = new HttpGet("http://account-service:8080/api/v1/account/getAllAccounts");
            CloseableHttpResponse responseBodyTask = httpclient.execute(httpgetTask);
            CloseableHttpResponse responseBodyEmployee = httpclient.execute(httpgetEmployee);
            CloseableHttpResponse responseBodyAccount = httpclient.execute(httpgetAccount);

            if (responseBodyTask.getStatusLine().getStatusCode() == 200) {
                String jsonContentTask = EntityUtils.toString(responseBodyTask.getEntity(), "UTF-8");
                JsonNode jsonNodeTask = objectMapper.readTree(jsonContentTask);

                String jsonContentEmployee = EntityUtils.toString(responseBodyEmployee.getEntity(), "UTF-8");
                JsonNode jsonNodeEmployee = objectMapper.readTree(jsonContentEmployee);

                String jsonContentAccount = EntityUtils.toString(responseBodyAccount.getEntity(), "UTF-8");
                JsonNode jsonNodeAccount = objectMapper.readTree(jsonContentAccount);

                Map<String, Object> item = new HashMap<>();
                int employeeId = jsonNodeTask.get("employeeId").asInt();

                // task management items
                item.put("id", jsonNodeTask.get("id"));
                item.put("incidentTitle", jsonNodeTask.get("incidentTitle"));
                item.put("incidentDesc", jsonNodeTask.get("incidentDesc"));
                item.put("incidentTimestamp", jsonNodeTask.get("incidentTimestamp"));
                item.put("severity", jsonNodeTask.get("severity"));
                item.put("status", jsonNodeTask.get("status"));
                item.put("dateAssigned", jsonNodeTask.get("dateAssigned"));
                item.put("accountId", jsonNodeTask.get("accountId"));
                item.put("truePositive", jsonNodeTask.get("truePositive"));
                item.put("logId", jsonNodeTask.get("logId"));

                // employee items

                item.put("employeeFirstname", jsonNodeEmployee.get(employeeId - 1).path("firstname"));
                item.put("employeeLastname", jsonNodeEmployee.get(employeeId - 1).path("lastname"));

                // soc account items
                if (jsonNodeTask.get("accountId").asInt() == 0) {
                    item.put("socName", null);
                } else {
                    item.put("socName", jsonNodeAccount.get(jsonNodeTask.get("accountId").asInt() - 1).path("name"));
                }


                httpclient.close();
                return ResponseEntity.ok(item);
            } else {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Task management list not found.");
            }
        } catch (HttpServerErrorException e) {
            return ResponseEntity.status(e.getStatusCode()).body("Error in retrieving task management list.");
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Internal server error.");
        }
    }

    @CrossOrigin(origins = "http://localhost:30008")
    @GetMapping("/viewAllCommentsByTaskId/{id}")
    public ResponseEntity<?> viewAllCommentsByTaskId(@PathVariable Long id) {
        try {
            objectMapper.findAndRegisterModules();
            CloseableHttpClient httpclient = HttpClients.createDefault();

            HttpGet httpgetComments = new HttpGet("http://comments-service:8083/api/v1/comments/taskManagement/" + id);
            HttpGet httpgetAccount = new HttpGet("http://account-service:8080/api/v1/account/getAllAccounts");
            CloseableHttpResponse responseBodyComments = httpclient.execute(httpgetComments);
            CloseableHttpResponse responseBodyAccount = httpclient.execute(httpgetAccount);

            // System.out.println(jsonNodeTask);
            if (responseBodyComments.getStatusLine().getStatusCode() == 200) {
                String jsonContentComments = EntityUtils.toString(responseBodyComments.getEntity(), "UTF-8");
                JsonNode jsonNodeComments = objectMapper.readTree(jsonContentComments);

                String jsonContentAccount = EntityUtils.toString(responseBodyAccount.getEntity(), "UTF-8");
                JsonNode jsonNodeAccount = objectMapper.readTree(jsonContentAccount);

                List<Map<String, Object>> commentsList = new ArrayList<>();
                for (int i = 0; i < jsonNodeComments.size(); i++) {
                    Map<String, Object> item = new HashMap<>();

                    // comment items
                    item.put("id", jsonNodeComments.get(i).path("id"));
                    item.put("commentDescription", jsonNodeComments.get(i).path("commentDescription"));

                    // soc account items
                    item.put("socName", jsonNodeAccount.get(jsonNodeComments.get("accountId").asInt() - 1).path("name"));

                    // Add the item to the list
                    commentsList.add(item);
                }

                httpclient.close();
                return ResponseEntity.ok(commentsList);
            } else {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Comments list not found.");
            }
        } catch (HttpServerErrorException e) {
            return ResponseEntity.status(e.getStatusCode()).body("Error in retrieving comments list.");
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Internal server error.");
        }
    }

    @CrossOrigin(origins = "http://localhost:30008")
    @GetMapping("/listSOCs")
    public ResponseEntity<?> listSOCs() {
        try {
            objectMapper.findAndRegisterModules();
            CloseableHttpClient httpclient = HttpClients.createDefault();

            HttpGet httpgetAccount = new HttpGet("http://account-service:8080/api/v1/account/getAllAccounts");
            CloseableHttpResponse responseBodyAccount = httpclient.execute(httpgetAccount);

            // System.out.println(jsonNodeTask);
            if (responseBodyAccount.getStatusLine().getStatusCode() == 200) {
                String jsonContentAccount = EntityUtils.toString(responseBodyAccount.getEntity(), "UTF-8");
                JsonNode jsonNodeAccount = objectMapper.readTree(jsonContentAccount);

                List<Map<String, Object>> socAccountList = new ArrayList<>();
                for (int i = 0; i < jsonNodeAccount.size(); i++) {
                    Map<String, Object> item = new HashMap<>();

                    item.put("id", jsonNodeAccount.get(i).path("id"));
                    item.put("socName", jsonNodeAccount.get(i).path("name"));

                    socAccountList.add(item);
                }

                httpclient.close();
                return ResponseEntity.ok(socAccountList);

            } else {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Account list not found.");
            }

        } catch (HttpServerErrorException e) {
            return ResponseEntity.status(e.getStatusCode()).body("Error in retrieving account list.");
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Internal server error.");
        }
    }

    @CrossOrigin(origins = "http://localhost:30008")
    @PutMapping("/assignSOC/{id}")
    public ResponseEntity<?> assignSOC(@PathVariable Long id, @RequestBody String newData) {
        try {
            objectMapper.findAndRegisterModules();
            CloseableHttpClient httpclient = HttpClients.createDefault();

            if (newData != null) {
                HttpPut httpPutNewAccountId = new HttpPut(
                        "http://task-management-service:8081/api/v1/accountUpdate/" + id);
                httpPutNewAccountId.setHeader("Accept", "application/json");
                httpPutNewAccountId.setHeader("Content-type", "application/json");
                StringEntity stringEntity = new StringEntity(newData);
                httpPutNewAccountId.setEntity(stringEntity);

                try (CloseableHttpResponse response = httpclient.execute(httpPutNewAccountId)) {

                    System.out.println("first api call and its status code" + response.getStatusLine().getStatusCode());

                    if (response.getEntity() == null) {
                        System.out.println("something went wrong here");
                    } else {
                        System.out.println("all good");
                    }

                } catch (IOException e) {
                    e.getMessage();
                }

                HttpPut httpPutNewStatus = new HttpPut("http://task-management-service:8081/api/v1/statusUpdate/" + id);
                httpPutNewStatus.setHeader("Accept", "application/json");
                httpPutNewStatus.setHeader("Content-type", "application/json");
                // StringEntity stringEntity2 = new StringEntity(newData);
                httpPutNewStatus.setEntity(stringEntity);

                try (CloseableHttpResponse response = httpclient.execute(httpPutNewStatus)) {
                    System.out
                            .println("second api call and its status code" + response.getStatusLine().getStatusCode());

                    if (response.getEntity() == null) {
                        System.out.println("something went wrong here");
                    } else {
                        System.out.println("all good");
                    }

                } catch (IOException e) {
                    e.getMessage();
                }
                HttpGet httpgetTask = new HttpGet("http://task-management-service:8081/api/v1/tasks/" + id);
                CloseableHttpResponse responseBodyTask = httpclient.execute(httpgetTask);
                System.out.println("this is responseBodyTask" + responseBodyTask);
                System.out.println(
                        "third api call and its status code" + responseBodyTask.getStatusLine().getStatusCode());
                if (responseBodyTask.getStatusLine().getStatusCode() == 200) {

                    String jsonContentTask = EntityUtils.toString(responseBodyTask.getEntity(), "UTF-8");
                    JsonNode jsonNodeTask = objectMapper.readTree(jsonContentTask);
                    String severity = jsonNodeTask.get("severity").toString();
                    String incidentTitle = jsonNodeTask.get("incidentTitle").toString();
                    String taskId = jsonNodeTask.get("id").toString();
                    JsonNode rootNode = objectMapper.readTree(newData);
                    Long accountId = rootNode.get("accountId").asLong();

                    CloseableHttpClient httpClient = HttpClients.createDefault();
                    HttpPost request1 = new HttpPost(
                            "http://notification-service:8080/api/v1/notifications/assignSOC/" + accountId);
                    Map<String, Object> requestBody = new HashMap<>();
                    requestBody.put("severity", severity);
                    requestBody.put("incidentTitle", incidentTitle);
                    requestBody.put("taskId", taskId);
                    ObjectMapper objectMapper = new ObjectMapper();
                    String json = objectMapper.writeValueAsString(requestBody);
                    StringEntity entity = new StringEntity(json, StandardCharsets.UTF_8);
                    request1.setEntity(entity);

                    Future<?> future = Executors.newSingleThreadExecutor().submit(() -> {
                        
                        try (CloseableHttpResponse response = httpClient.execute(request1)) {
                        System.out.println(EntityUtils.toString(response.getEntity()));
                        System.out.println("fourth api call and its status code" +
                        response.getStatusLine().getStatusCode());
                        } catch (Exception e) {
                        e.printStackTrace();
                        }
                    });
                }
                return ResponseEntity.ok(newData);

            } else {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Unable to assign.");
            }
        } catch (HttpServerErrorException e) {
            return ResponseEntity.status(e.getStatusCode()).body("Error in assigning SOC.");
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Internal server error.");
        }
    }

    @CrossOrigin(origins = "http://localhost:30008")
    @PutMapping("/changeStatus/{id}")
    public ResponseEntity<?> changeStatus(@PathVariable Long id, @RequestBody String newData) {
        try {
            objectMapper.findAndRegisterModules();
            CloseableHttpClient httpclient = HttpClients.createDefault();

            if (newData != null) {
                HttpPut httpPutNewStatus = new HttpPut("http://task-management-service:8081/api/v1/statusUpdate/" + id);
                httpPutNewStatus.setHeader("Accept", "application/json");
                httpPutNewStatus.setHeader("Content-type", "application/json");
                StringEntity stringEntity = new StringEntity(newData);
                httpPutNewStatus.setEntity(stringEntity);

                try (CloseableHttpResponse response = httpclient.execute(httpPutNewStatus)) {

                    if (response.getEntity() == null) {
                        System.out.println("something went wrong here");
                    } else {
                        System.out.println("all good");
                    }

                } catch (IOException e) {
                    e.getMessage();
                }

                httpclient.close();
                return ResponseEntity.ok(newData);

            } else {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Unable to assign.");
            }
        } catch (HttpServerErrorException e) {
            return ResponseEntity.status(e.getStatusCode()).body("Error in assigning SOC.");
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Internal server error.");
        }
    }

    @CrossOrigin(origins = "http://localhost:30008")
    @PutMapping("/closeCase/{id}")
    public ResponseEntity<?> closeCase(@PathVariable Long id, @RequestBody String newData) {
        try {
            objectMapper.findAndRegisterModules();
            CloseableHttpClient httpclient = HttpClients.createDefault();

            HttpGet httpgetTask = new HttpGet("http://task-management-service:8081/api/v1/tasks/" + id);
            CloseableHttpResponse responseBodyTask = httpclient.execute(httpgetTask);
            String jsonContentTask = EntityUtils.toString(responseBodyTask.getEntity(), "UTF-8");
            JsonNode jsonNodeTask = objectMapper.readTree(jsonContentTask);

            if (newData != null) {
                HttpPut httpPutNewRiskRating = new HttpPut(
                        "http://behavioral-analysis-service:8084/api/v1/updateRiskRating/"
                                + jsonNodeTask.get("employeeId"));
                httpPutNewRiskRating.setHeader("Accept", "application/json");
                httpPutNewRiskRating.setHeader("Content-type", "application/json");
                StringEntity stringEntity = new StringEntity(newData); // send 0 in json if no change, send new risk
                                                                       // rating in json if there is a change
                httpPutNewRiskRating.setEntity(stringEntity);

                try (CloseableHttpResponse response = httpclient.execute(httpPutNewRiskRating)) {

                    if (response.getEntity() == null) {
                        System.out.println("something went wrong here");
                    } else {
                        System.out.println("all good");
                    }

                } catch (IOException e) {
                    e.getMessage();
                }

                httpclient.close();
                return ResponseEntity.ok(newData);

            } else {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Unable to assign.");
            }
        } catch (HttpServerErrorException e) {
            return ResponseEntity.status(e.getStatusCode()).body("Error in assigning SOC.");
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Internal server error.");
        }
    }

    @CrossOrigin(origins = "http://localhost:30008")
    @GetMapping("/viewAllTasksByEmployeeId/{id}")
    public ResponseEntity<?> viewAllTasksByEmployeeId(@PathVariable int id) {
        try {
            objectMapper.findAndRegisterModules();
            CloseableHttpClient httpclient = HttpClients.createDefault();

            HttpGet httpgetTask = new HttpGet("http://task-management-service:8081/api/v1/tasks/employee/" + id);
            HttpGet httpgetAccount = new HttpGet("http://account-service:8080/api/v1/account/getAllAccounts");
            CloseableHttpResponse responseBodyAccount = httpclient.execute(httpgetAccount);
            CloseableHttpResponse responseBodyTask = httpclient.execute(httpgetTask);

            // System.out.println(jsonNodeTask);
            if (responseBodyTask.getStatusLine().getStatusCode() == 200) {
                String jsonContentTask = EntityUtils.toString(responseBodyTask.getEntity(), "UTF-8");
                JsonNode jsonNodeTask = objectMapper.readTree(jsonContentTask);

                String jsonContentAccount = EntityUtils.toString(responseBodyAccount.getEntity(), "UTF-8");
                JsonNode jsonNodeAccount = objectMapper.readTree(jsonContentAccount);

                List<Map<String, Object>> taskManagementList = new ArrayList<>();
                for (int i = 0; i < jsonNodeTask.size(); i++) {
                    Map<String, Object> item = new HashMap<>();

                    // task management items
                    item.put("id", jsonNodeTask.get(i).path("id"));
                    item.put("incidentTitle", jsonNodeTask.get(i).path("incidentTitle"));
                    item.put("incidentDesc", jsonNodeTask.get(i).path("incidentDesc"));
                    item.put("incidentTimestamp", jsonNodeTask.get(i).path("incidentTimestamp"));
                    item.put("severity", jsonNodeTask.get(i).path("severity"));
                    item.put("status", jsonNodeTask.get(i).path("status"));
                    item.put("dateAssigned", jsonNodeTask.get(i).path("dateAssigned"));
                    item.put("truePositive", jsonNodeTask.get(i).path("truePositive"));
                    item.put("logId", jsonNodeTask.get(i).path("logId"));

                    // soc account items
                    if (jsonNodeTask.get(i).path("accountId").asInt() == 0) {
                        item.put("socName", null);
                    } else {
                        item.put("socName", jsonNodeAccount.get(jsonNodeTask.get(i).path("accountId").asInt() - 1).path("name"));
                    }
                    

                    // Add the item to the list
                    taskManagementList.add(item);
                }

                httpclient.close();
                return ResponseEntity.ok(taskManagementList);
            } else {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Task management list not found.");
            }
        } catch (HttpServerErrorException e) {
            return ResponseEntity.status(e.getStatusCode()).body("Error in retrieving task management list.");
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Internal server error.");
        }
    }

    @Async
    @CrossOrigin(origins = "http://localhost:30008")
    @PostMapping("/addComment/{id}")
    public ResponseEntity<?> addComment(@PathVariable Long id, @RequestBody String newComment) {
        try {
            objectMapper.findAndRegisterModules();
            CloseableHttpClient httpclient = HttpClients.createDefault();
            HttpGet httpgetTask = new HttpGet("http://task-management-service:8081/api/v1/tasks/" + id); // task id
            CloseableHttpResponse responseBodyTask = httpclient.execute(httpgetTask);

            if (responseBodyTask != null) {
                HttpPost httpPostComment = new HttpPost("http://comments-service:8083/api/v1" + "/comments");
                StringEntity stringEntity = new StringEntity(newComment);
                httpPostComment.setEntity(stringEntity);
                //httpclient.execute(httpPostComment);

                try (CloseableHttpResponse response = httpclient.execute(httpPostComment);) {

                    if (response.getEntity() == null) {
                    System.out.println("something went wrong here");
                    } else {
                    //System.out.println("all good");
                    }

                } catch (IOException e) {
                    e.getMessage();
                }

                httpclient.close();
                return ResponseEntity.ok(newComment);
            } else {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Task not found.");
            }
        } catch (HttpServerErrorException e) {
            return ResponseEntity.status(e.getStatusCode()).body("Error in retrieving task.");
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Internal server error.");
        }
    }

}
