package com.service.rulebasedalgorithm;

import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.springframework.stereotype.Service;

@Service
public class RuleBasedAlgorithmService {

    CloseableHttpClient httpclient = HttpClients.createDefault();
    ObjectMapper objectMapper = new ObjectMapper();
    
    public String checkCaseSix (int bytesOut, int id) throws ClientProtocolException, IOException {
        objectMapper.findAndRegisterModules();
        String task = "";

        if (bytesOut >= 10000000) {

            HttpGet httpgetEmployee = new HttpGet("http://employee-service:8082/api/v1/employees" + "/" + id);

            ObjectNode obj = objectMapper.createObjectNode();
            
            try (CloseableHttpResponse response =  httpclient.execute(httpgetEmployee)) {

                if (response.getEntity() != null) {
                    String jsonContentEmployee = EntityUtils.toString(response.getEntity(), "UTF-8");
                    JsonNode jsonNodeEmployee = objectMapper.readTree(jsonContentEmployee);
                    obj.put("incidentDesc", jsonNodeEmployee.get("firstname").asText() + " " + jsonNodeEmployee.get("lastname").asText() + " has made an unusually large amount of data upload or download.");
                } else {
                    System.out.println("something went wrong here");
                }

            } catch (IOException e) {
                e.getMessage();
            }

            obj.put("incidentTitle", "Potential Data Exfiltration");
            obj.put("severity", 100);
            obj.put("accountId", 0);
            obj.put("employeeId", id);
            task = obj.toString();

        } 

        return task;
    }

    public String checkCaseFour (String status, int id) throws ClientProtocolException, IOException {
        objectMapper.findAndRegisterModules();
        String task = "";
     
        if (status.equals("FAIL")) {

            HttpGet httpgetEmployee = new HttpGet("http://employee-service:8082/api/v1/employees" + "/" + id);

            ObjectMapper objectMapper = new ObjectMapper();
            ObjectNode obj = objectMapper.createObjectNode();
            
            try (CloseableHttpResponse response =  httpclient.execute(httpgetEmployee)) {

                if (response.getEntity() != null) {
                    String jsonContentEmployee = EntityUtils.toString(response.getEntity(), "UTF-8");
                    JsonNode jsonNodeEmployee = objectMapper.readTree(jsonContentEmployee);
                    obj.put("incidentDesc", jsonNodeEmployee.get("firstname").asText() + " " + jsonNodeEmployee.get("lastname").asText() + " failed to enter the building.");
                } else {
                    System.out.println("something went wrong here");
                }

            } catch (IOException e) {
                e.getMessage();
            }

            obj.put("incidentTitle", "Failed Attempt to Enter Building");
            obj.put("severity", 50);
            obj.put("accountId", 0);
            obj.put("employeeId", id);
            task = obj.toString();

        } 

        return task;
    }

    public String checkCaseFive (int id, String officeLocation) throws ClientProtocolException, IOException {
        objectMapper.findAndRegisterModules();
        String task = "";
     
        HttpGet httpgetEmployee = new HttpGet("http://employee-service:8082/api/v1/employees" + "/" + id);
        CloseableHttpResponse responseBodyEmployee = httpclient.execute(httpgetEmployee);

        if (responseBodyEmployee.getEntity() != null) {

            String jsonContentEmployee = EntityUtils.toString(responseBodyEmployee.getEntity(), "UTF-8");
            JsonNode jsonNodeEmployee = objectMapper.readTree(jsonContentEmployee);

            if (!(officeLocation.equals(jsonNodeEmployee.get("location").asText()))) {

                ObjectMapper objectMapper = new ObjectMapper();
                ObjectNode obj = objectMapper.createObjectNode();

                obj.put("incidentTitle", "Impossible Traveller");
                obj.put("incidentDesc", jsonNodeEmployee.get("firstname").asText() + " " + jsonNodeEmployee.get("lastname").asText() + " has logged on or off from a machine at a location different from their own. This raises concerns of a potential account compromise or a malicious actor impersonating the employee.");
                obj.put("severity", 200);
                obj.put("accountId", 0);
                obj.put("employeeId", id);
                task = obj.toString();

            } 
        }
        return task;
    }

    public String checkCaseThree (String logOnOff, int id) throws ClientProtocolException, IOException, ParseException {
        objectMapper.findAndRegisterModules();
        String task = "";

        if (logOnOff.equals("Log On")) {
            HttpGet httpgetEmployee = new HttpGet("http://employee-service:8082/api/v1/employees" + "/" + id);
            CloseableHttpResponse responseBodyEmployee = httpclient.execute(httpgetEmployee);

            if (responseBodyEmployee.getEntity() != null) {

                String jsonContentEmployee = EntityUtils.toString(responseBodyEmployee.getEntity(), "UTF-8");
                JsonNode jsonNodeEmployee = objectMapper.readTree(jsonContentEmployee);

                if (jsonNodeEmployee.get("terminatedDate") == null) {

                    ObjectMapper objectMapper = new ObjectMapper();
                    ObjectNode obj = objectMapper.createObjectNode();

                    obj.put("incidentTitle", "Terminated Employee Login");
                    obj.put("incidentDesc", jsonNodeEmployee.get("firstname").asText() + " " + jsonNodeEmployee.get("lastname").asText() + " has logged on or off from a machine post termination.");
                    obj.put("severity", 200);
                    obj.put("accountId", 0);
                    obj.put("employeeId", id);
                    task = obj.toString();
                    
                } 
            } 
        }
        
        return task;
    }

    public String checkCaseOne (String accessDateTime, int id, int workingHours) throws ClientProtocolException, IOException, ParseException {
        objectMapper.findAndRegisterModules();
        String task = "";

        String original = accessDateTime;
        String timestampString = original.replace("T", " ");
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

        try {
            Date timestamp = dateFormat.parse(timestampString);
            Calendar cal = Calendar.getInstance();
            cal.setTime(timestamp);

            int dayOfWeek = cal.get(Calendar.DAY_OF_WEEK);
            int hour = cal.get(Calendar.HOUR_OF_DAY);

            // Profile 1: 9am - 6pm Mon-Fri
            if (dayOfWeek >= Calendar.MONDAY && dayOfWeek <= Calendar.FRIDAY && hour >= 8 && hour <= 19 && workingHours == 1) {
                task = "";
            }
            // Profile 2: 9am - 6pm any day
            else if (hour >= 8 && hour <= 19 && workingHours == 2) {
                task = "";
            }
            // Profile 3: 12-hour shift (8am-8pm or 8pm-8am) any day
            else if (hour >= 7 && hour <= 21 && workingHours == 3) {
                task = "";
            }
            else if (workingHours == 4 && ((hour >= 19 && hour <= 24) || (hour >= 0 && hour <= 9))) {
                task = "";
            }
            else {
                HttpGet httpgetEmployee = new HttpGet("http://employee-service:8082/api/v1/employees" + "/" + id);

                ObjectMapper objectMapper = new ObjectMapper();
                ObjectNode obj = objectMapper.createObjectNode();
                
                try (CloseableHttpResponse response =  httpclient.execute(httpgetEmployee)) {

                    if (response.getEntity() != null) {
                        String jsonContentEmployee = EntityUtils.toString(response.getEntity(), "UTF-8");
                        JsonNode jsonNodeEmployee = objectMapper.readTree(jsonContentEmployee);
                        obj.put("incidentDesc", jsonNodeEmployee.get("firstname").asText() + " " + jsonNodeEmployee.get("lastname").asText() + " has attempted to log into their account after working hours.");
                    } else {
                        System.out.println("something went wrong here");
                    }

                } catch (IOException e) {
                    e.getMessage();
                }

                obj.put("incidentTitle", "After Hour Login");
                obj.put("severity", 50);
                obj.put("accountId", 0);
                obj.put("employeeId", id);
                task = obj.toString();
            }
        } catch (ParseException e) {
            System.err.println("Invalid timestamp format");
        }

        return task;
    }

    public String checkCaseTwo (String accessDateTimePC, int id, String machineLocation) throws ClientProtocolException, IOException, ParseException {
        objectMapper.findAndRegisterModules();
        String task = "";
        
        HttpGet httpgetEmployee = new HttpGet("http://employee-service:8082/api/v1/employees" + "/" + id);
        CloseableHttpResponse responseBodyEmployee = httpclient.execute(httpgetEmployee);

        if (responseBodyEmployee.getEntity() != null) {

            String jsonContentEmployee = EntityUtils.toString(responseBodyEmployee.getEntity(), "UTF-8");
            JsonNode jsonNodeEmployee = objectMapper.readTree(jsonContentEmployee);

            if (!(machineLocation.equals(jsonNodeEmployee.get("location").asText()))) {

                HttpGet httpgetBALog = new HttpGet("http://building-access:8087/api/v1/buildingaccesslogs");
                CloseableHttpResponse responseBodyBALog = httpclient.execute(httpgetBALog);
                String jsonContentBA = EntityUtils.toString(responseBodyBALog.getEntity(), "UTF-8");
                JsonNode jsonNodeBA = objectMapper.readTree(jsonContentBA);

                for (int j = 0; j < jsonNodeBA.size(); j++) {
                    String timestamp1 = accessDateTimePC.replace("T", " ");
                    String timestamp2 = jsonNodeBA.get(j).path("accessDateTime").asText().replace("T", " ");;
                    SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
                    // Parse the first timestamp
                    Date timestampPC = dateFormat.parse(timestamp1);
                    
                    // Parse the second timestamp
                    Date timestampBA = dateFormat.parse(timestamp2);

                    long timeDifferenceMillis = Math.abs(timestampPC.getTime() - timestampBA.getTime());
                    
                    if (timeDifferenceMillis <= 20 * 60 * 1000) {
                        if (jsonNodeBA.get(j).path("userId").asInt() == id && jsonNodeBA.get(j).path("officeLocation").asText().equals(machineLocation) && (!jsonNodeBA.get(j).path("officeLocation").asText().equals(jsonNodeEmployee.get("location").asText()))) {
                            task = "case 5";
                            // System.out.println("Debug: Time difference is within 20 minutes.");
                            // System.out.println("Debug: UserId match: " + jsonNodeBA.get(j).path("userId").asInt() + "=" + id);
                            // System.out.println("Debug: Office location match: " + jsonNodeBA.get(j).path("officeLocation").asText() + "=" + machineLocation);
                            // System.out.println("Debug: Employee location mismatch: " + jsonNodeBA.get(j).path("officeLocation").asText() + " != " + jsonNodeEmployee.get("location").asText());
                            return task;
                        }
                    } 
                }

                ObjectMapper objectMapper = new ObjectMapper();
                ObjectNode obj = objectMapper.createObjectNode();

                obj.put("incidentTitle", "Potential Account Sharing");
                obj.put("incidentDesc", jsonNodeEmployee.get("firstname").asText() + " " + jsonNodeEmployee.get("lastname").asText() + "'s account may be compromised.");
                obj.put("severity", 100);
                obj.put("accountId", 0);
                obj.put("employeeId", id);
                task = obj.toString();

            }
        }
        return task;
    }

    
}
