package com.service.commentsservice;

import java.util.*;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface CommentsServiceRepository extends JpaRepository<CommentsService, Long> {
    
    List<CommentsService> findByTaskManagementId(Long taskManagementId);

    List<CommentsService> findByAccountId(Long accountId);


}
