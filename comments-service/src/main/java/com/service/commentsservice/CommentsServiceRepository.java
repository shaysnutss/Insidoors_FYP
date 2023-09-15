package com.service.commentsservice;

import java.util.*;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface CommentsServiceRepository extends JpaRepository<CommentsService, Long> {
    
    List<CommentsService> findAllByTaskManagementId(int taskManagementId);

    List<CommentsService> findAllByAccountId(int accountId);

}
