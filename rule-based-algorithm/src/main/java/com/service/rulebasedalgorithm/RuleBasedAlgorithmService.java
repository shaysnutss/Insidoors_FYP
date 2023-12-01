package com.service.rulebasedalgorithm;

import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;

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
    
    public boolean checkCaseSix (int bytesOut) throws ClientProtocolException, IOException {

        if (bytesOut >= 10000000) {
            return true;
        } 

        return false;
    }

    public boolean checkCaseFive (String officeLocation, String employeeLocation) throws ClientProtocolException, IOException {

        if (officeLocation != null && employeeLocation != null) {

            if (!(officeLocation.equals(employeeLocation))) {
                return true;
            } 
        }
        return false;
    }

    public boolean checkCaseFour (String status) throws ClientProtocolException, IOException {
     
        if (status.equals("FAIL")) {
            return true;
        } 

        return false;
    }

    public boolean checkCaseThree (String terminatedDate) throws ClientProtocolException, IOException, ParseException {

        if (!terminatedDate.equals("null") && LocalDate.parse(terminatedDate, DateTimeFormatter.ISO_DATE).isBefore(LocalDate.now())) {
            //System.out.println(terminatedDate);
           return true;
        }
        
        return false;
    }

    public boolean checkCaseOne (String accessDateTime, int workingHours) throws ClientProtocolException, IOException, ParseException {

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
                return false;
            }
            // Profile 2: 9am - 6pm any day
            else if (hour >= 8 && hour <= 19 && workingHours == 2) {
                return false;
            }
            // Profile 3: 12-hour shift (8am-8pm or 8pm-8am) any day
            else if (hour >= 7 && hour <= 21 && workingHours == 3) {
                return false;
            }
            else if (workingHours == 4 && ((hour >= 19 && hour <= 24) || (hour >= 0 && hour <= 9))) {
                return false;
            }

            
        } catch (ParseException e) {
            System.err.println("Invalid timestamp format");
        }
        return true;
    }

    public int checkCaseTwo (String accessDateTimePC, int id, String machineLocation, String employeeLocation, JsonNode jsonNodeBA) throws ClientProtocolException, IOException, ParseException {

        if (!(machineLocation.equals(employeeLocation))) {

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
                    if (jsonNodeBA.get(j).path("userId").asInt() == id && jsonNodeBA.get(j).path("officeLocation").asText().equals(machineLocation) && (!jsonNodeBA.get(j).path("officeLocation").asText().equals(employeeLocation))) {
                        return 5;
                    }
                } 
            }

            return 2;

        } else {
            return 0;
        }
        
    }

    public ObjectNode makeCase7RequestBody(JsonNode item, JsonNode jsonNodeEmployee) {
        ObjectNode obj = objectMapper.createObjectNode();
        obj.put("id", item.path("id").asInt());
        obj.put("incidentDesc", jsonNodeEmployee.get(item.path("user_id").asInt() - 1).path("firstname").asText() + " " + jsonNodeEmployee.get(item.path("user_id").asInt() - 1).path("lastname").asText() + " is tied to an unknown case.");
        obj.put("incidentTitle", "Potential Security Breach");
        obj.put("incidentTimestamp", item.path("accessDateTime").asText().replace("T", " "));
        obj.put("severity", 50);
        obj.put("accountId", 0);
        obj.put("employeeId", item.path("user_id").asInt());
        obj.put("suspect", 7);
        return obj;
    }

    public ObjectNode makeCase6RequestBody(JsonNode taskPrePost, JsonNode jsonNodeEmployee) {
        ObjectNode obj = objectMapper.createObjectNode();
        obj.put("id", taskPrePost.path("id").asInt());
        obj.put("incidentDesc", jsonNodeEmployee.get(taskPrePost.path("userId").asInt() - 1).path("firstname").asText() + " " + jsonNodeEmployee.get(taskPrePost.path("userId").asInt() - 1).path("lastname").asText() + " has made an unusually large amount of data upload or download.");
        obj.put("incidentTitle", "Potential Data Exfiltration");
        obj.put("incidentTimestamp", taskPrePost.path("accessDateTime").asText().replace("T", " "));
        obj.put("severity", 100);
        obj.put("accountId", 0);
        obj.put("employeeId", taskPrePost.path("userId").asInt());
        obj.put("suspect", 6);
        return obj;
    }

    public ObjectNode makeCase5RequestBody(JsonNode taskPrePost, JsonNode jsonNodeEmployee) {
        ObjectNode obj = objectMapper.createObjectNode();
        obj.put("id", taskPrePost.path("id").asInt());
        obj.put("incidentTimestamp", taskPrePost.path("accessDateTime").asText().replace("T", " "));
        obj.put("incidentDesc", jsonNodeEmployee.get(taskPrePost.path("userId").asInt() - 1).path("firstname").asText() + " " + jsonNodeEmployee.get(taskPrePost.path("userId").asInt() - 1).path("lastname").asText() + " has logged on or off from a machine at a location different from their own. This raises concerns of a potential account compromise or a malicious actor impersonating the employee.");
        obj.put("incidentTitle", "Impossible Traveller");
        obj.put("severity", 200);
        obj.put("accountId", 0);
        obj.put("employeeId", taskPrePost.path("userId").asInt());
        obj.put("suspect", 5);
        return obj;
    }

    public ObjectNode makeCase4RequestBody(JsonNode taskPrePost, JsonNode jsonNodeEmployee) {
        ObjectNode obj = objectMapper.createObjectNode();
        obj.put("id", taskPrePost.path("id").asInt());
        obj.put("incidentDesc", jsonNodeEmployee.get(taskPrePost.path("userId").asInt() - 1).path("firstname").asText() + " " + jsonNodeEmployee.get(taskPrePost.path("userId").asInt() - 1).path("lastname").asText() + " failed to enter the building.");
        obj.put("incidentTitle", "Failed Attempt to Enter Building");
        obj.put("incidentTimestamp", taskPrePost.path("accessDateTime").asText().replace("T", " "));
        obj.put("severity", 50);
        obj.put("accountId", 0);
        obj.put("employeeId", taskPrePost.path("userId").asInt());
        obj.put("suspect", 4);
        return obj;
    }

    public ObjectNode makeCase3RequestBody(JsonNode taskPrePost, JsonNode jsonNodeEmployee) {
        ObjectNode obj = objectMapper.createObjectNode();
        obj.put("id", taskPrePost.path("id").asInt());
        obj.put("incidentDesc", jsonNodeEmployee.get(taskPrePost.path("userId").asInt() - 1).path("firstname").asText() + " " + jsonNodeEmployee.get(taskPrePost.path("userId").asInt() - 1).path("lastname").asText() + " has logged on or off from a machine post termination.");
        obj.put("incidentTitle", "Terminated Employee Login");
        obj.put("incidentTimestamp", taskPrePost.path("accessDateTime").asText().replace("T", " "));
        obj.put("severity", 200);
        obj.put("accountId", 0);
        obj.put("employeeId", taskPrePost.path("userId").asInt());
        obj.put("suspect", 3);
        return obj;
    }

    public ObjectNode makeCase2RequestBody(JsonNode taskPrePost, JsonNode jsonNodeEmployee) {
        ObjectNode obj = objectMapper.createObjectNode();
        obj.put("id", taskPrePost.path("id").asInt());
        obj.put("incidentDesc", jsonNodeEmployee.get(taskPrePost.path("userId").asInt() - 1).path("firstname").asText() + " " + jsonNodeEmployee.get(taskPrePost.path("userId").asInt() - 1).path("lastname").asText() + "'s account may be compromised.");
        obj.put("incidentTitle", "Potential Account Sharing");
        obj.put("severity", 100);
        obj.put("accountId", 0);
        obj.put("employeeId", taskPrePost.path("userId").asInt());
        obj.put("suspect", 2);
        obj.put("incidentTimestamp", taskPrePost.path("accessDateTime").asText().replace("T", " "));
        return obj;
    }

    public ObjectNode makeCase1RequestBody(JsonNode taskPrePost, JsonNode jsonNodeEmployee) {
        ObjectNode obj = objectMapper.createObjectNode();
        obj.put("id", taskPrePost.path("id").asInt());
        obj.put("incidentDesc", jsonNodeEmployee.get(taskPrePost.path("userId").asInt() - 1).path("firstname").asText() + " " + jsonNodeEmployee.get(taskPrePost.path("userId").asInt() - 1).path("lastname").asText() + " has attempted to log into their account after working hours.");
        obj.put("incidentTitle", "After Hour Login");
        obj.put("severity", 50);
        obj.put("accountId", 0);
        obj.put("employeeId", taskPrePost.path("userId").asInt());
        obj.put("suspect", 1);
        obj.put("incidentTimestamp", taskPrePost.path("accessDateTime").asText().replace("T", " "));
        return obj;
    }

    public ObjectNode pcObject(JsonNode jsonNode, JsonNode jsonNodeEmployee, int employeeId, int i) {
        ObjectNode obj = objectMapper.createObjectNode();
        obj.put("id", jsonNode.get(i).path("id").asInt());
        obj.put("user_id", jsonNode.get(i).path("userId").asInt());
        obj.put("access_date_time", jsonNode.get(i).path("accessDateTime").asText());
        obj.put("log_on_off", jsonNode.get(i).path("logOnOff").asText());
        obj.put("machine_name", jsonNode.get(i).path("machineName").asText());
        obj.put("machine_location", jsonNode.get(i).path("machineLocation").asText());
        obj.put("working_hours", jsonNode.get(i).path("workingHours").asInt());
        obj.put("user_location", jsonNodeEmployee.get(employeeId - 1).path("location").asText());
        obj.put("terminated", jsonNodeEmployee.get(employeeId - 1).path("terminatedDate").asText().equals("null") ? "N" : "Y");
        obj.put("suspect", jsonNode.get(i).path("suspect").asInt());
        return obj;
    }

    public ObjectNode buildingObject(JsonNode jsonNode, int i, JsonNode jsonNodeEmployee, int employeeId) {
        ObjectNode obj = objectMapper.createObjectNode();
        obj.put("id", jsonNode.get(i).path("id").asInt());
        obj.put("user_id", jsonNode.get(i).path("userId").asInt());
        obj.put("access_date_time", jsonNode.get(i).path("accessDateTime").asText());
        obj.put("direction", jsonNode.get(i).path("direction").asText());
        obj.put("status", jsonNode.get(i).path("status").asText());
        obj.put("office_location", jsonNode.get(i).path("office_location").asText());
        obj.put("suspect", jsonNode.get(i).path("suspect").asInt());
        obj.put("attempts", jsonNode.get(i).path("attempts").asInt());
        obj.put("terminated", jsonNodeEmployee.get(employeeId - 1).path("terminatedDate").asText().equals("null") ? "N" : "Y");
        return obj;
    }

    public ObjectNode proxyObject(JsonNode jsonNode, int i) {
        ObjectNode obj = objectMapper.createObjectNode();
        obj.put("id", jsonNode.get(i).path("id").asInt());
        obj.put("user_id", jsonNode.get(i).path("userId").asInt());
        obj.put("access_date_time", jsonNode.get(i).path("accessDateTime").asText());
        obj.put("machine_name", jsonNode.get(i).path("machineName").asText());
        obj.put("url", jsonNode.get(i).path("url").asText());
        obj.put("category", jsonNode.get(i).path("category").asText());
        obj.put("bytes_in", jsonNode.get(i).path("bytesIn").asInt());
        obj.put("bytes_out", jsonNode.get(i).path("bytesOut").asInt());
        obj.put("suspect", jsonNode.get(i).path("suspect").asInt());
        return obj;
    }
}
