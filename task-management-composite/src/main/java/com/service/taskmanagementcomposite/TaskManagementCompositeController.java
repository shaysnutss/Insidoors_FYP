package com.service.taskmanagementcomposite;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.CompletableFuture;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.apache.http.client.config.RequestConfig;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPut;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.HttpServerErrorException;

@RestController
@RequestMapping(path = "/api/v1")
public class TaskManagementCompositeController {

    CloseableHttpClient httpclient = HttpClients.createDefault();
    ObjectMapper objectMapper = new ObjectMapper();
    
    @GetMapping("/viewAllTasks")
    public ResponseEntity<?> viewAllTasks() {
        try {
            objectMapper.findAndRegisterModules();

            HttpGet httpgetTask = new HttpGet("http://task-management-service:8081/api/v1/tasks");
            CloseableHttpResponse responseBodyTask = httpclient.execute(httpgetTask);

            if (responseBodyTask.getStatusLine().getStatusCode() == 200) {
                String jsonContentTask = EntityUtils.toString(responseBodyTask.getEntity(), "UTF-8");
                JsonNode jsonNodeTask = objectMapper.readTree(jsonContentTask);
                
                List<Map<String, Object>> taskManagementList = new ArrayList<>();
                for (int i = 0; i < jsonNodeTask.size(); i++) {
                    Map<String, Object> item = new HashMap<>();

                    // task management items
                    item.put("id", jsonNodeTask.get(i).path("id"));
                    item.put("incidentTitle", jsonNodeTask.get(i).path("incidentTitle"));
                    item.put("incidentTimestamp", jsonNodeTask.get(i).path("incidentTimestamp"));
                    item.put("severity", jsonNodeTask.get(i).path("severity"));
                    item.put("status", jsonNodeTask.get(i).path("status"));
                    item.put("dateAssigned", jsonNodeTask.get(i).path("dateAssigned"));

                    // employee items
                    HttpGet httpgetEmployee = new HttpGet("http://employee-service:8082/api/v1/employees/" + jsonNodeTask.get(i).path("employeeId"));
                    CloseableHttpResponse responseBodyEmployee = httpclient.execute(httpgetEmployee);
                    String jsonContentEmployee = EntityUtils.toString(responseBodyEmployee.getEntity(), "UTF-8");
                    JsonNode jsonNodeEmployee = objectMapper.readTree(jsonContentEmployee);

                    item.put("employeeFirstname", jsonNodeEmployee.get("firstname"));
                    item.put("employeeLastname", jsonNodeEmployee.get("lastname"));
                    
                    // soc account items
                    HttpGet httpgetAccount = new HttpGet("http://account-service:8080/api/v1/account/getAccountById/" + jsonNodeTask.get(i).path("accountId"));
                    CloseableHttpResponse responseBodyAccount = httpclient.execute(httpgetAccount);
                    String jsonContentAccount = EntityUtils.toString(responseBodyAccount.getEntity(), "UTF-8");
                    JsonNode jsonNodeAccount = objectMapper.readTree(jsonContentAccount);

                    item.put("socName", jsonNodeAccount.get("name"));

                    // Add the item to the list
                    taskManagementList.add(item);
                }
                
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

    @GetMapping("/viewAllTasksByAccountId/{id}")
    public ResponseEntity<?> viewAllTasksByAccountId(@PathVariable int id) {
        try {
            objectMapper.findAndRegisterModules();

            HttpGet httpgetTask = new HttpGet("http://task-management-service:8081/api/v1/tasks/account/" + id);
            CloseableHttpResponse responseBodyTask = httpclient.execute(httpgetTask);

            //System.out.println(jsonNodeTask);
            if (responseBodyTask.getStatusLine().getStatusCode() == 200) {
                String jsonContentTask = EntityUtils.toString(responseBodyTask.getEntity(), "UTF-8");
                JsonNode jsonNodeTask = objectMapper.readTree(jsonContentTask);
                
                List<Map<String, Object>> taskManagementList = new ArrayList<>();
                for (int i = 0; i < jsonNodeTask.size(); i++) {
                    Map<String, Object> item = new HashMap<>();

                    // task management items
                    item.put("id", jsonNodeTask.get(i).path("id"));
                    item.put("incidentTitle", jsonNodeTask.get(i).path("incidentTitle"));
                    item.put("incidentTimestamp", jsonNodeTask.get(i).path("incidentTimestamp"));
                    item.put("severity", jsonNodeTask.get(i).path("severity"));
                    item.put("status", jsonNodeTask.get(i).path("status"));
                    item.put("dateAssigned", jsonNodeTask.get(i).path("dateAssigned"));

                    // employee items
                    HttpGet httpgetEmployee = new HttpGet("http://employee-service:8082/api/v1/employees/" + jsonNodeTask.get(i).path("employeeId"));
                    CloseableHttpResponse responseBodyEmployee = httpclient.execute(httpgetEmployee);
                    String jsonContentEmployee = EntityUtils.toString(responseBodyEmployee.getEntity(), "UTF-8");
                    JsonNode jsonNodeEmployee = objectMapper.readTree(jsonContentEmployee);

                    item.put("employeeFirstname", jsonNodeEmployee.get("firstname"));
                    item.put("employeeLastname", jsonNodeEmployee.get("lastname"));
                    
                    // soc account items
                    HttpGet httpgetAccount = new HttpGet("http://account-service:8080/api/v1/account/getAccountById/" + jsonNodeTask.get(i).path("accountId"));
                    CloseableHttpResponse responseBodyAccount = httpclient.execute(httpgetAccount);
                    String jsonContentAccount = EntityUtils.toString(responseBodyAccount.getEntity(), "UTF-8");
                    JsonNode jsonNodeAccount = objectMapper.readTree(jsonContentAccount);

                    item.put("socName", jsonNodeAccount.get("name"));

                    // Add the item to the list
                    taskManagementList.add(item);
                }
                
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

    @GetMapping("/viewAllCommentsByTaskId/{id}")
    public ResponseEntity<?> viewAllCommentsByTaskId(@PathVariable Long id) {
        try {
            objectMapper.findAndRegisterModules();

            HttpGet httpgetComments = new HttpGet("http://comments-service:8083/api/v1/comments/taskManagement/" + id);
            CloseableHttpResponse responseBodyComments = httpclient.execute(httpgetComments);

            //System.out.println(jsonNodeTask);
            if (responseBodyComments.getStatusLine().getStatusCode() == 200) {
                String jsonContentComments = EntityUtils.toString(responseBodyComments.getEntity(), "UTF-8");
                JsonNode jsonNodeComments = objectMapper.readTree(jsonContentComments);

                List<Map<String, Object>> commentsList = new ArrayList<>();
                for (int i = 0; i < jsonNodeComments.size(); i++) {
                    Map<String, Object> item = new HashMap<>();

                    // comment items
                    item.put("id", jsonNodeComments.get(i).path("id"));
                    item.put("commentDescription", jsonNodeComments.get(i).path("commentDescription"));
                    
                    // soc account items
                    HttpGet httpgetAccount = new HttpGet("http://account-service:8080/api/v1/account/getAccountById/" + jsonNodeComments.get(i).path("accountId"));
                    CloseableHttpResponse responseBodyAccount = httpclient.execute(httpgetAccount);
                    String jsonContentAccount = EntityUtils.toString(responseBodyAccount.getEntity(), "UTF-8");
                    JsonNode jsonNodeAccount = objectMapper.readTree(jsonContentAccount);

                    item.put("socName", jsonNodeAccount.get("name"));

                    // Add the item to the list
                    commentsList.add(item);
                }
                
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

    @GetMapping("/listSOCs")
    public ResponseEntity<?> listSOCs() {
        try {
            objectMapper.findAndRegisterModules();

            HttpGet httpgetAccount = new HttpGet("http://account-service:8080/api/v1/account/getAllAccounts");
            CloseableHttpResponse responseBodyAccount = httpclient.execute(httpgetAccount);

            //System.out.println(jsonNodeTask);
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

    @PutMapping("/assignSOC/{id}")
    public ResponseEntity<?> assignSOC(@PathVariable Long id, @RequestBody String newData) {
        try {
            objectMapper.findAndRegisterModules();

            if (newData != null) {
                HttpPut httpPutNewAccountId = new HttpPut("http://task-management-service:8081/api/v1/accountUpdate/" + id);
                httpPutNewAccountId.setHeader("Accept", "application/json");
                httpPutNewAccountId.setHeader("Content-type", "application/json");
                StringEntity stringEntity = new StringEntity(newData);
                httpPutNewAccountId.setEntity(stringEntity);

                RequestConfig requestConfig = RequestConfig.custom()
                    .setConnectTimeout(5000) // Connection timeout in milliseconds
                    .setSocketTimeout(5000)  // Socket timeout in milliseconds
                    .build();
                httpPutNewAccountId.setConfig(requestConfig);

                // Perform the PUT request asynchronously
                CompletableFuture<CloseableHttpResponse> accountResponseFuture = CompletableFuture.supplyAsync(() -> {
                    try {
                        return httpclient.execute(httpPutNewAccountId);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                });

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

    @PutMapping("/changeStatus/{id}")
    public ResponseEntity<?> changeStatus(@PathVariable Long id, @RequestBody String newData) {
        try {
            objectMapper.findAndRegisterModules();

            if (newData != null) {
                HttpPut httpPutNewStatus = new HttpPut("http://task-management-service:8081/api/v1/statusUpdate/" + id);
                httpPutNewStatus.setHeader("Accept", "application/json");
                httpPutNewStatus.setHeader("Content-type", "application/json");
                StringEntity stringEntity = new StringEntity(newData);
                httpPutNewStatus.setEntity(stringEntity);

                RequestConfig requestConfig = RequestConfig.custom()
                    .setConnectTimeout(5000) // Connection timeout in milliseconds
                    .setSocketTimeout(5000)  // Socket timeout in milliseconds
                    .build();
                httpPutNewStatus.setConfig(requestConfig);

                // Perform the PUT request asynchronously
                CompletableFuture<CloseableHttpResponse> statusResponseFuture = CompletableFuture.supplyAsync(() -> {
                    try {
                        return httpclient.execute(httpPutNewStatus);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                });

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

    @PutMapping("/closeCase/{id}")
    public ResponseEntity<?> closeCase(@PathVariable Long id, @RequestBody String newData) {
        try {
            objectMapper.findAndRegisterModules();

            HttpGet httpgetTask = new HttpGet("http://task-management-service:8081/api/v1/tasks/" + id);
            CloseableHttpResponse responseBodyTask = httpclient.execute(httpgetTask);
            String jsonContentTask = EntityUtils.toString(responseBodyTask.getEntity(), "UTF-8");
            JsonNode jsonNodeTask = objectMapper.readTree(jsonContentTask);
            
            if (newData != null) {
                HttpPut httpPutNewRiskRating = new HttpPut("http://behavioral-analysis-service:8084/api/v1/updateRiskRating/" + jsonNodeTask.get("employeeId"));
                httpPutNewRiskRating.setHeader("Accept", "application/json");
                httpPutNewRiskRating.setHeader("Content-type", "application/json");
                StringEntity stringEntity = new StringEntity(newData); // send 0 in json if no change, send new risk rating in json if there is a change
                httpPutNewRiskRating.setEntity(stringEntity);

                RequestConfig requestConfig = RequestConfig.custom()
                    .setConnectTimeout(5000) // Connection timeout in milliseconds
                    .setSocketTimeout(5000)  // Socket timeout in milliseconds
                    .build();
                httpPutNewRiskRating.setConfig(requestConfig);

                // Perform the PUT request asynchronously
                CompletableFuture<CloseableHttpResponse> ratingResponseFuture = CompletableFuture.supplyAsync(() -> {
                    try {
                        return httpclient.execute(httpPutNewRiskRating);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                });

                HttpPut httpPutStatus = new HttpPut("http://task-management-service:8081/api/v1/statusUpdate/" + id);
                httpPutStatus.setHeader("Accept", "application/json");
                httpPutStatus.setHeader("Content-type", "application/json");
                httpPutStatus.setEntity(stringEntity); // json must have status closed

                httpPutStatus.setConfig(requestConfig);

                // Perform the PUT request asynchronously
                CompletableFuture<CloseableHttpResponse> statusResponseFuture = CompletableFuture.supplyAsync(() -> {
                    try {
                        return httpclient.execute(httpPutStatus);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                });
                

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

}
